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
#   "python-dotenv"
# ]
# ///

"""
main.py
--------------------------------------------------------------------------------
Hugging Face Job: QLoRA Finetuning Experiment (KTO - Preference Optimization)
--------------------------------------------------------------------------------
Author: The Neural Maze
Context: Lecture on LLM Finetuning Techniques.

DESCRIPTION:
This script performs QLoRA (Quantized Low-Rank Adaptation) finetuning using 
Kahneman-Tversky Optimization (KTO). KTO is a preference optimization method
that, unlike DPO, does not require paired preferences (chosen/rejected). 
Instead, it uses a binary "desirable" vs "undesirable" signal.

KEY CONCEPTS DEMONSTRATED:
1. QLoRA (Quantized LoRA): Load the base model in 4-bit NF4 format.
2. KTO (Kahneman-Tversky Optimization): Aligning the model with human preferences
   using the KTOTrainer.
3. Model Merging: Merging adapters back into the base model.

CLI INSTRUCTION
hf jobs uv run main.py `
    --flavor a10g-small `
    --timeout 1h `
    --max_steps 1000 `
    --num_train_epochs 1 `
    -e COMET_PROJECT_NAME="finetuning-sessions-lab5" `
    -s COMET_API_KEY="XXXX" `
    -s HF_TOKEN="XXXX"
"""

import sys
import logging as log
import comet_ml
import torch
import fire
import os
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import KTOTrainer, KTOConfig
from transformers import TrainingArguments
from dotenv import load_dotenv

# Load environment variables (e.g. for HF_TOKEN and COMET_API_KEY)
load_dotenv()

# --- LOGGING SETUP ---
root = log.getLogger()
root.setLevel(log.INFO)
handler = log.StreamHandler(sys.stdout)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


def main(
    # --- MODEL PARAMETERS ---
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct", 
    load_in_4bit: bool = True,

    # --- LORA PARAMETERS ---
    lora_r: int = 16, 
    lora_alpha: int = 16, 
    lora_dropout: float = 0.0, 
    
    # --- DATASET PARAMETERS ---
    dataset_name: str = "trl-lib/kto-mix-14k",
    dataset_num_rows: int = None, 
    eval_num_rows: int = None, 

    # --- TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "hedrergudene/Qwen2.5-1.5B-KTO-Finetuning",
    max_seq_length: int = 4096,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-7, 
    num_train_epochs: int = 1,
    max_steps: int = -1,
    beta: float = 0.1, # KTO beta hyperparameter
) -> None:
    
    log.info("================================================================")
    log.info("       STARTING QLORA FINETUNING EXPERIMENT (KTO)              ")
    log.info("================================================================")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD MODEL & ADD LORA ADAPTERS (QLORA)
    # --------------------------------------------------------------------------
    log.info(f"Loading Base Model in 4-bit (QLoRA): {model_name}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None, 
        load_in_4bit=load_in_4bit, 
    )

    log.info("Adding LoRA adapters to the quantized model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    log.info("Model prepared for QLoRA training.")

    # --------------------------------------------------------------------------
    # STEP 2: PREPARE DATASET
    # --------------------------------------------------------------------------
    log.info(f"Loading dataset: {dataset_name}")
    train_split = "train" if dataset_num_rows is None else f"train[:{dataset_num_rows}]"
    dataset = load_dataset(dataset_name, split=train_split)

    def format_kto(example):
        # 1. Format Prompt
        if isinstance(example["prompt"], list):
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            example["prompt"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": str(example["prompt"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            
        # 2. Format Completion: Extract string content safely
        completion = example.get("completion", "")
        
        if isinstance(completion, list):
            if len(completion) > 0:
                first_item = completion[0]
                # Case: [{'role': 'assistant', 'content': '...'}]
                if isinstance(first_item, dict):
                    example["completion"] = str(first_item.get("content", ""))
                # Case: ['Just a string inside a list']
                else:
                    example["completion"] = str(first_item)
            else:
                example["completion"] = ""
        else:
            # Case: Already a string or None
            example["completion"] = str(completion) if completion is not None else ""
        
        return example

    dataset = dataset.map(format_kto)

    # --- DEBUG: Verify types before Trainer starts ---
    sample_prompt = dataset[0]["prompt"]
    sample_completion = dataset[0]["completion"]
    log.info(f"Check - Prompt type: {type(sample_prompt)}, Completion type: {type(sample_completion)}")
    
    if not isinstance(sample_prompt, str) or not isinstance(sample_completion, str):
        raise TypeError(f"Dataset formatting failed! Expected strings, got {type(sample_prompt)} and {type(sample_completion)}")

    # --------------------------------------------------------------------------
    # STEP 3: CONFIGURE TRAINING
    # --------------------------------------------------------------------------
    training_args = KTOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        beta=beta,
        
        # Optimization settings
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.1,
        
        # Logging & Saving
        logging_steps=1,
        save_strategy="no", 
        report_to=["comet_ml"],
        seed=3407,
        remove_unused_columns=False,
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # --------------------------------------------------------------------------
    # STEP 4: EXECUTE TRAINING
    # --------------------------------------------------------------------------
    log.info("Starting Training Loop...")
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = torch.cuda.memory_allocated() / 1024**3
    
    log.info(f"GPU: {gpu_stats.name}")
    log.info(f"Total VRAM Available: {gpu_stats.total_memory / 1024**3:.2f} GB")
    log.info(f"Pre-Train VRAM Used: {start_gpu_memory:.2f} GB")
    
    trainer_stats = trainer.train()
    
    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    log.info(f"Training Complete. Final Loss: {trainer_stats.training_loss:.4f}")
    log.info(f"Peak VRAM Used: {peak_gpu_memory:.2f} GB")

    # --------------------------------------------------------------------------
    # STEP 5: SAVE & PUSH FULL MODEL (MERGED)
    # --------------------------------------------------------------------------
    log.info(f"Merging and pushing FULL model to Hugging Face Hub: {hub_model_id}...")
    
    model.push_to_hub_merged(
        hub_model_id, 
        tokenizer, 
        save_method = "merged_16bit",
        # token = os.environ.get("HF_TOKEN"), # Unsloth picks this up automatically from env
    )
    
    log.info("Push complete!")

if __name__ == "__main__":
    fire.Fire(main)