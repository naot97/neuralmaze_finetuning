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
# ]
# ///

"""
main.py
--------------------------------------------------------------------------------
Hugging Face Job: QLoRA Vision Finetuning (Qwen3-VL-2B LaTeX OCR)
--------------------------------------------------------------------------------
Author: The Neural Maze
Context: Lecture on LLM Finetuning Techniques.

DESCRIPTION:
This script performs QLoRA finetuning on Qwen3-VL-2B-Instruct for LaTeX OCR —
converting images of handwritten math formulas into LaTeX code. It uses
Unsloth's FastVisionModel with 4-bit quantization and LoRA adapters on both
vision and language layers.

KEY CONCEPTS DEMONSTRATED:
1. Vision-Language Model (VLM) finetuning with QLoRA.
2. FastVisionModel: Unsloth's optimized loader for vision models.
3. UnslothVisionDataCollator: Required data collator for vision finetuning.
4. Multi-modal input: images + text instructions in chat format.

CLI INSTRUCTION
hf jobs uv run main.py `
    --flavor a10g-small `
    --timeout 1h `
    --max_steps 30 `
    -e COMET_PROJECT_NAME="finetuning-sessions-lab6a" `
    -s COMET_API_KEY="XXXX" `
    -s HF_TOKEN="XXXX"
"""

import sys
import logging as log

import comet_ml
import torch
import fire
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
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
    model_name: str = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    load_in_4bit: bool = True,

    # --- LORA PARAMETERS ---
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,

    # --- DATASET PARAMETERS ---
    dataset_name: str = "unsloth/LaTeX_OCR",
    instruction: str = "Write the LaTeX representation for this image.",

    # --- TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "naot97/Qwen3-VL-2B-LaTeX-OCR-Finetuning",
    max_seq_length: int = 2048,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    max_steps: int = 30,
    warmup_steps: int = 5,
    report_to: str = "comet_ml",
) -> None:

    log.info("================================================================")
    log.info("   STARTING QLORA VISION FINETUNING (Qwen3-VL-2B LaTeX OCR)   ")
    log.info("================================================================")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD MODEL & ADD LORA ADAPTERS (QLoRA)
    # --------------------------------------------------------------------------
    log.info(f"Loading Vision Model in 4-bit (QLoRA): {model_name}...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    log.info("Adding LoRA adapters to vision + language layers...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    log.info("Model prepared for QLoRA vision finetuning.")

    # --------------------------------------------------------------------------
    # STEP 2: PREPARE DATASET
    # --------------------------------------------------------------------------
    log.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["text"]},
                ],
            },
        ]
        return {"messages": conversation}

    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    log.info(f"Dataset size: {len(converted_dataset)} samples")

    # --------------------------------------------------------------------------
    # STEP 3: CONFIGURE TRAINING
    # --------------------------------------------------------------------------
    FastVisionModel.for_training(model)

    training_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to=[report_to],

        # Required for vision finetuning
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=training_args,
    )

    # --------------------------------------------------------------------------
    # STEP 4: EXECUTE TRAINING
    # --------------------------------------------------------------------------
    log.info("Starting Training Loop...")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = torch.cuda.max_memory_reserved() / 1024**3

    log.info(f"GPU: {gpu_stats.name}")
    log.info(f"Total VRAM Available: {gpu_stats.total_memory / 1024**3:.2f} GB")
    log.info(f"Pre-Train VRAM Reserved: {start_gpu_memory:.2f} GB")

    trainer_stats = trainer.train()

    peak_gpu_memory = torch.cuda.max_memory_reserved() / 1024**3
    used_for_training = peak_gpu_memory - start_gpu_memory

    log.info(f"Training Complete. Final Loss: {trainer_stats.training_loss:.4f}")
    log.info(f"Training Time: {trainer_stats.metrics['train_runtime']:.1f}s")
    log.info(f"Peak VRAM Reserved: {peak_gpu_memory:.2f} GB")
    log.info(f"VRAM Used for Training: {used_for_training:.2f} GB")

    # --------------------------------------------------------------------------
    # STEP 5: SAVE & PUSH MODEL
    # --------------------------------------------------------------------------
    log.info(f"Pushing merged model to Hugging Face Hub: {hub_model_id}...")

    model.push_to_hub_merged(
        hub_model_id,
        tokenizer,
        save_method="merged_16bit",
    )

    log.info("Push complete!")


if __name__ == "__main__":
    fire.Fire(main)
