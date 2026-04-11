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
#   "snac",
#   "torchaudio",
# ]
# ///

"""
main.py
--------------------------------------------------------------------------------
Hugging Face Job: LoRA Finetuning Experiment (Orpheus-3B TTS)
--------------------------------------------------------------------------------
Author: The Neural Maze
Context: Lecture on LLM Finetuning Techniques.

DESCRIPTION:
This script finetunes the Orpheus-3B text-to-speech model using LoRA. Orpheus
treats TTS as a language modeling task — text is tokenized normally, and audio
is encoded into discrete SNAC codec tokens. The model learns to generate audio
token sequences from text input.

KEY CONCEPTS DEMONSTRATED:
1. LoRA finetuning on a TTS language model (Orpheus-3B).
2. SNAC audio codec: encoding waveforms into discrete token sequences.
3. Custom tokenization: interleaving text tokens with special TTS control tokens
   (start_of_human, end_of_human, start_of_ai, start_of_speech, etc.).
4. Duplicate frame removal for cleaner audio output.

CLI INSTRUCTION
hf jobs uv run main.py `
    --flavor a10g-small `
    --timeout 1h `
    --max_steps 60 `
    -e COMET_PROJECT_NAME="finetuning-sessions-lab6b" `
    -s COMET_API_KEY="XXXX" `
    -s HF_TOKEN="XXXX"
"""

import sys
import locale
import logging as log

import comet_ml
import torch
import torchaudio.transforms as T
import fire
from snac import SNAC
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from dotenv import load_dotenv

# Load environment variables (e.g. for HF_TOKEN and COMET_API_KEY)
load_dotenv()

locale.getpreferredencoding = lambda: "UTF-8"

# --- LOGGING SETUP ---
root = log.getLogger()
root.setLevel(log.INFO)
handler = log.StreamHandler(sys.stdout)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# --- SPECIAL TOKEN IDS ---
TOKENISER_LENGTH = 128256
START_OF_TEXT    = 128000
END_OF_TEXT      = 128009
START_OF_SPEECH  = TOKENISER_LENGTH + 1
END_OF_SPEECH    = TOKENISER_LENGTH + 2
START_OF_HUMAN   = TOKENISER_LENGTH + 3
END_OF_HUMAN     = TOKENISER_LENGTH + 4
START_OF_AI      = TOKENISER_LENGTH + 5
END_OF_AI        = TOKENISER_LENGTH + 6
PAD_TOKEN        = TOKENISER_LENGTH + 7
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10


def tokenise_audio(waveform, snac_model, ds_sample_rate):
    """Encode a waveform into SNAC discrete token codes."""
    waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

    return all_codes


def remove_duplicate_frames(example):
    """Remove consecutive duplicate SNAC frames (groups of 7 tokens)."""
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]
    for i in range(7, len(vals), 7):
        if vals[i] != result[-7]:
            result.extend(vals[i:i + 7])

    example["codes_list"] = result
    return example


def main(
    # --- MODEL PARAMETERS ---
    model_name: str = "unsloth/orpheus-3b-0.1-ft",
    load_in_4bit: bool = False,
    max_seq_length: int = 2048,

    # --- LORA PARAMETERS ---
    lora_r: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,

    # --- DATASET PARAMETERS ---
    dataset_name: str = "MrDragonFox/Elise",

    # --- TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "naot97/Orpheus-3B-TTS-Finetuning",
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    max_steps: int = 60,
    warmup_steps: int = 5,
    report_to: str = "comet_ml",
) -> None:

    log.info("================================================================")
    log.info("     STARTING LORA FINETUNING EXPERIMENT (Orpheus-3B TTS)      ")
    log.info("================================================================")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD MODEL & ADD LORA ADAPTERS
    # --------------------------------------------------------------------------
    log.info(f"Loading Base Model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    log.info("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    log.info("Model prepared with LoRA adapters.")

    # --------------------------------------------------------------------------
    # STEP 2: PREPARE DATASET (SNAC TOKENIZATION)
    # --------------------------------------------------------------------------
    log.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    ds_sample_rate = dataset[0]["audio"]["sampling_rate"]

    log.info("Loading SNAC codec model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

    log.info("Tokenizing audio with SNAC codec...")

    def add_codes(example):
        codes_list = None
        try:
            answer_audio = example.get("audio")
            if answer_audio and "array" in answer_audio:
                codes_list = tokenise_audio(
                    answer_audio["array"], snac_model, ds_sample_rate
                )
        except Exception as e:
            print(f"Skipping row due to error: {e}")
        example["codes_list"] = codes_list
        return example

    dataset = dataset.map(add_codes, remove_columns=["audio"])

    # Filter invalid rows and remove duplicate frames
    dataset = dataset.filter(lambda x: x["codes_list"] is not None)
    dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)
    dataset = dataset.map(remove_duplicate_frames)

    log.info(f"Dataset size after filtering: {len(dataset)} samples")

    # Create input_ids with special tokens
    def create_input_ids(example):
        text_prompt = (
            f"{example['source']}: {example['text']}"
            if "source" in example else example["text"]
        )

        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)

        input_ids = (
            [START_OF_HUMAN]
            + text_ids
            + [END_OF_HUMAN]
            + [START_OF_AI]
            + [START_OF_SPEECH]
            + example["codes_list"]
            + [END_OF_SPEECH]
            + [END_OF_AI]
        )
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        return example

    dataset = dataset.map(
        create_input_ids, remove_columns=["text", "codes_list"]
    )

    # Keep only required columns
    columns_to_remove = [
        col for col in dataset.column_names
        if col not in ["input_ids", "labels", "attention_mask"]
    ]
    dataset = dataset.remove_columns(columns_to_remove)

    # Free SNAC model from GPU
    snac_model.to("cpu")
    torch.cuda.empty_cache()

    log.info("Dataset preparation complete.")

    # --------------------------------------------------------------------------
    # STEP 3: CONFIGURE TRAINING
    # --------------------------------------------------------------------------
    training_args = TrainingArguments(
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
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
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
    log.info(f"Merging and pushing model to Hugging Face Hub: {hub_model_id}...")

    model.push_to_hub_merged(
        hub_model_id,
        tokenizer,
        save_method="merged_16bit",
    )

    log.info("Push complete!")


if __name__ == "__main__":
    fire.Fire(main)
