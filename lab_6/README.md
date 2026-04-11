# Lab 5b: GRPO Finetuning (Qwen3-4B Reasoning & GPT-OSS-20B Code Optimization)

This repository contains scripts for GRPO (Group Relative Policy Optimization) finetuning experiments using Unsloth.

Two experiments are included:
- **Qwen3-4B**: Train a math reasoning model with custom reasoning format (SFT pre-finetuning + GRPO)
- **GPT-OSS-20B**: Train a code optimization model to write fast pure-Python matrix multiplication (GRPO only)

## Setup

### Using `pip`

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Using `Docker` (Recommended)

If you're having trouble with large libraries like `unsloth` or specialized CUDA requirements, you can run the lab inside our pre-configured container:

**Run with GPU support**:

To enter the container and run commands manually:
```bash
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  -e COMET_API_KEY="your-key" \
  -e HF_TOKEN="your-token" \
  theneuralmaze/finetuning-sessions bash
```

## Usage

### Training

**Qwen3-4B GRPO** (math reasoning with SFT pre-finetuning + GRPO):
```bash
python main.py
```

**GPT-OSS-20B GRPO** (code optimization):
```bash
python main_gpt_oss.py
```

### Inference

**Qwen3-4B** reasoning model:
```bash
python inference.py qwen3 --prompt "What is the sqrt of 101?"
```

**GPT-OSS-20B** code optimization model:
```bash
python inference.py gpt_oss
```

Note: This requires a GPU and sufficient VRAM to load the model.
