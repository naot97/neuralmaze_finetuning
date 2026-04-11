# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is "The Neural Maze" finetuning labs — a series of self-contained labs that progressively teach LLM finetuning techniques using Unsloth and Hugging Face tooling. Each `lab_N/` directory is an independent project with its own `pyproject.toml`, `main.py`, and `inference.py`.

## Lab Progression

- **lab_0**: Environment diagnostics — verifies CUDA, Unsloth, and GPU availability
- **lab_1**: Continued Pretraining (CPT) — full finetuning on raw text (math domain adaptation)
- **lab_2**: Full Supervised Finetuning (SFT) — instruction tuning with 100% weight updates
- **lab_3**: LoRA finetuning (SFT) — parameter-efficient fine-tuning with low-rank adapters
- **lab_4**: QLoRA finetuning (SFT) — 4-bit quantized LoRA for lower memory usage
- **lab_5**: QLoRA + KTO — preference optimization using Kahneman-Tversky Optimization
- **lab_6**: GRPO Finetuning (Qwen3-4B Reasoning & GPT-OSS-20B Code Optimization)
- **lab_7a**: QLoRA Vision Finetuning (Qwen3-VL-8B LaTeX OCR)
- **lab_7b**: LoRA Finetuning (Orpheus-3B TTS with SNAC codec)

## Running Labs

Each lab uses Python Fire for CLI arguments and can run locally or via Docker.

```bash
# Local (inside lab directory, requires GPU + CUDA)
python main.py                          # uses default parameters
python main.py --model_name "..." --learning_rate 1e-5  # override defaults

# Docker (recommended — handles CUDA/Unsloth dependencies)
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  -e COMET_API_KEY="..." \
  -e HF_TOKEN="..." \
  theneuralmaze/finetuning-sessions bash

# Inference (after training)
python inference.py --model_name "model-id" --prompt "your prompt"

# HF Jobs remote execution (used in lab docstrings)
hf jobs uv run main.py --flavor a10g-small
```

## Key Architecture Patterns

- **Inline script dependencies**: Each `main.py` declares dependencies in PEP 723 `# /// script` blocks for `uv run` compatibility on HF Jobs
- **Unsloth's `FastLanguageModel`**: All labs use this as the model loader — it wraps HF transformers with optimized kernels and Flash Attention 2
- **TRL trainers**: Training uses `SFTTrainer` (labs 1-4, 7a-7b), `KTOTrainer` (lab 5), and `GRPOTrainer` (lab 6) from the `trl` library
- **Comet ML**: Experiment tracking via `report_to=["comet_ml"]` in training args
- **`unsloth_compiled_cache/`**: Auto-generated cache directories — not source code, do not edit

## Environment Variables

- `HF_TOKEN`: Hugging Face auth token (for model push/pull)
- `COMET_API_KEY` / `COMET_PROJECT_NAME`: Experiment tracking
- `HF_ENDPOINT_URL` / `HF_API_TOKEN`: For inference against deployed endpoints (lab_0)
