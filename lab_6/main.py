#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "comet_ml",
#   "unsloth",
#   "transformers",
#   "datasets",
#   "trl",
#   "huggingface_hub",
#   "python-dotenv",
#   "vllm",
#   "pandas",
#   "numpy",
# ]
# ///

"""
main.py
--------------------------------------------------------------------------------
Hugging Face Job: GRPO Finetuning Experiment (Qwen3-4B Reasoning)
--------------------------------------------------------------------------------
Author: The Neural Maze
Context: Lecture on LLM Finetuning Techniques.

DESCRIPTION:
This script converts Qwen3-4B-Base into a reasoning model via GRPO (Group
Relative Policy Optimization). The process has two phases:
1. Pre-finetuning (SFT): Teach the model a custom reasoning format using a
   small subset of NVIDIA's OpenMathReasoning dataset.
2. GRPO Training: Train the model with multiple reward functions on the
   DAPO-Math-17k dataset to improve mathematical reasoning.

KEY CONCEPTS DEMONSTRATED:
1. LoRA finetuning with vLLM fast inference for generation during GRPO.
2. Custom chat template with reasoning tags (<start_working_out>, <end_working_out>).
3. Multi-reward GRPO: format matching, approximate format, answer checking, number extraction.
4. Two-phase training: SFT pre-finetuning → GRPO preference optimization.

CLI INSTRUCTION
hf jobs uv run main.py `
    --flavor a10g-small `
    --timeout 2h `
    --max_steps 100 `
    -e COMET_PROJECT_NAME="finetuning-sessions-lab5b" `
    -s COMET_API_KEY="XXXX" `
    -s HF_TOKEN="XXXX"
"""

import sys
import os
import re
import gc
import logging as log

import comet_ml
import torch
import fire
import numpy as np
import pandas as pd
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
from dotenv import load_dotenv

# Load environment variables (e.g. for HF_TOKEN and COMET_API_KEY)
load_dotenv()

# Disable expandable segments (conflicts with vLLM standby)
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.pop("UNSLOTH_VLLM_STANDBY", None)

# --- LOGGING SETUP ---
root = log.getLogger()
root.setLevel(log.INFO)
handler = log.StreamHandler(sys.stdout)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# --- REASONING FORMAT TAGS ---
REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)


def build_chat_template():
    """Build a custom chat template with reasoning start prepended on generation."""
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
        "{% else %}"
            "{{ '" + SYSTEM_PROMPT + "' + eos_token }}"
            "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '" + REASONING_START + "' }}"
        "{% endif %}"
    )
    return chat_template


def build_reward_functions(tokenizer):
    """Build regex patterns and reward functions for GRPO training."""

    # Regex for matching the full format
    solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"

    match_format = re.compile(
        rf"{REASONING_END}.*?"
        rf"{SOLUTION_START}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )

    # Regex for extracting numbers from solutions
    match_numbers = re.compile(
        SOLUTION_START + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags=re.MULTILINE | re.DOTALL
    )

    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            score += 0.5 if response.count(REASONING_END)  == 1 else -1.0
            score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
            score += 0.5 if response.count(SOLUTION_END)   == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := match_format.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-2.0)
                continue
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if   ratio >= 0.9 and ratio <= 1.1: score += 2.0
                    elif ratio >= 0.8 and ratio <= 1.2: score += 1.5
                    else: score -= 2.5
                except:
                    score -= 4.5
            scores.append(score)
        return scores

    printed_times = {"count": 0}
    print_every_steps = 5

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        if printed_times["count"] % print_every_steps == 0:
            print(
                '*' * 20 + f"Question:\n{question}",
                f"\nAnswer:\n{answer[0]}",
                f"\nResponse:\n{responses[0]}",
                f"\nExtracted:\n{extracted_responses[0]}"
            )
        printed_times["count"] += 1

        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess == true_answer else -1.5)
            except:
                scores.append(0)
                continue
        return scores

    return [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]


def main(
    # --- MODEL PARAMETERS ---
    model_name: str = "unsloth/Qwen3-0.6B-Base",
    load_in_4bit: bool = False,
    fast_inference: bool = True,
    offload_embedding: bool = True,
    gpu_memory_utilization: float = 0.6,

    # --- LORA PARAMETERS ---
    lora_r: int = 32,
    lora_alpha: int = 64,

    # --- DATASET PARAMETERS ---
    sft_dataset_name: str = "unsloth/OpenMathReasoning-mini",
    grpo_dataset_name: str = "open-r1/DAPO-Math-17k-Processed",
    grpo_dataset_config: str = "en",

    # --- PRE-FINETUNING (SFT) PARAMETERS ---
    sft_batch_size: int = 1,
    sft_gradient_accumulation_steps: int = 1,
    sft_learning_rate: float = 2e-4,
    sft_num_train_epochs: int = 1,
    sft_warmup_steps: int = 5,

    # --- GRPO TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "naot97/Qwen3-0.6B-GRPO-Finetuning",
    max_seq_length: int = 2048,
    grpo_batch_size: int = 1,
    grpo_gradient_accumulation_steps: int = 1,
    grpo_learning_rate: float = 5e-6,
    grpo_max_steps: int = 100,
    num_generations: int = 4,
    prompt_length_quantile: float = 0.9,
    report_to: str = "comet_ml",
) -> None:

    log.info("================================================================")
    log.info("       STARTING GRPO FINETUNING EXPERIMENT (Qwen3-4B)          ")
    log.info("================================================================")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD MODEL & ADD LORA ADAPTERS
    # --------------------------------------------------------------------------
    log.info(f"Loading Base Model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        offload_embedding=offload_embedding,
        max_lora_rank=lora_r,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    log.info("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Set custom chat template
    tokenizer.chat_template = build_chat_template()
    log.info("Model prepared with LoRA adapters and custom chat template.")

    # --------------------------------------------------------------------------
    # STEP 2: PRE-FINETUNING (SFT) FOR FORMAT LEARNING
    # --------------------------------------------------------------------------
    log.info(f"Loading SFT dataset: {sft_dataset_name}")

    sft_dataset = load_dataset(sft_dataset_name, split="cot")
    sft_df = sft_dataset.to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]

    # Filter to numeric answers only
    is_number = pd.to_numeric(
        pd.Series(sft_df["expected_answer"]), errors="coerce"
    ).notnull()
    sft_df = sft_df.iloc[np.where(is_number)[0]]

    def format_sft_dataset(x):
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")
        thoughts = thoughts.strip()
        final_prompt = (
            REASONING_START + thoughts + REASONING_END +
            SOLUTION_START + x["expected_answer"] + SOLUTION_END
        )
        return [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": x["problem"]},
            {"role": "assistant", "content": final_prompt},
        ]

    sft_df["Messages"] = sft_df.apply(format_sft_dataset, axis=1)

    # Truncate to max_seq_length/2 to avoid overly long reasoning traces
    sft_df["N"] = sft_df["Messages"].apply(
        lambda x: len(tokenizer.apply_chat_template(x))
    )
    sft_df = sft_df.loc[sft_df["N"] <= max_seq_length / 2].copy()
    log.info(f"SFT dataset size after filtering: {sft_df.shape[0]}")

    sft_df["text"] = tokenizer.apply_chat_template(
        sft_df["Messages"].values.tolist(), tokenize=False
    )
    sft_hf_dataset = Dataset.from_pandas(sft_df)

    log.info("Starting SFT pre-finetuning for format learning...")
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_hf_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=sft_batch_size,
            gradient_accumulation_steps=sft_gradient_accumulation_steps,
            warmup_steps=sft_warmup_steps,
            num_train_epochs=sft_num_train_epochs,
            learning_rate=sft_learning_rate,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )
    sft_trainer.train()
    log.info("SFT pre-finetuning complete.")

    # Free SFT dataset memory
    del sft_dataset, sft_df, sft_hf_dataset, sft_trainer
    torch.cuda.empty_cache()
    gc.collect()

    # --------------------------------------------------------------------------
    # STEP 3: PREPARE GRPO DATASET
    # --------------------------------------------------------------------------
    log.info(f"Loading GRPO dataset: {grpo_dataset_name}")

    dataset = load_dataset(grpo_dataset_name, grpo_dataset_config, split="train")

    def extract_hash_answer(text):
        return text

    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    })

    # Filter out top 10% longest prompts
    tokenized = dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        },
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    maximum_length = int(np.quantile(tokenized["L"], prompt_length_quantile))
    log.info(f"Max prompt length (p{int(prompt_length_quantile*100)}): {maximum_length}")

    dataset = dataset.select(
        np.where(np.array(tokenized["L"]) <= maximum_length)[0]
    )
    del tokenized
    log.info(f"GRPO dataset size after filtering: {len(dataset)}")

    # --------------------------------------------------------------------------
    # STEP 4: BUILD REWARD FUNCTIONS & CONFIGURE GRPO
    # --------------------------------------------------------------------------
    reward_funcs = build_reward_functions(tokenizer)

    max_prompt_length = maximum_length + 1
    max_completion_length = max_seq_length - max_prompt_length

    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=grpo_learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=grpo_batch_size,
        gradient_accumulation_steps=grpo_gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=grpo_max_steps,
        save_steps=grpo_max_steps,
        report_to=[report_to],
        output_dir=output_dir,
        seed=3407,
    )

    # --------------------------------------------------------------------------
    # STEP 5: EXECUTE GRPO TRAINING
    # --------------------------------------------------------------------------
    log.info("Starting GRPO Training Loop...")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = torch.cuda.memory_allocated() / 1024**3

    log.info(f"GPU: {gpu_stats.name}")
    log.info(f"Total VRAM Available: {gpu_stats.total_memory / 1024**3:.2f} GB")
    log.info(f"Pre-Train VRAM Used: {start_gpu_memory:.2f} GB")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer_stats = trainer.train()

    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3

    log.info(f"GRPO Training Complete. Final Loss: {trainer_stats.training_loss:.4f}")
    log.info(f"Peak VRAM Used: {peak_gpu_memory:.2f} GB")

    # --------------------------------------------------------------------------
    # STEP 6: SAVE & PUSH FULL MODEL (MERGED)
    # --------------------------------------------------------------------------
    log.info(f"Merging and pushing FULL model to Hugging Face Hub: {hub_model_id}...")

    model.push_to_hub_merged(
        hub_model_id,
        tokenizer,
        save_method="merged_16bit",
    )

    log.info("Push complete!")


if __name__ == "__main__":
    fire.Fire(main)
