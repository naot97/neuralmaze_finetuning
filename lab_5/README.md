# Lab 5: QLoRA Finetuning (KTO - Preference Optimization)

This repository contains scripts for performing Parameter-Efficient Fine-Tuning (PEFT) using QLoRA (Quantized Low-Rank Adaptation) on small language models using Unsloth.

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

To run the entrypoint script (`main.py`) directly:
```bash
docker run --rm --gpus all \
  -v $(pwd):/app \
  -e COMET_API_KEY="your-key" \
  -e HF_TOKEN="your-token" \
  theneuralmaze/finetuning-sessions
```

## Usage

### Training

To start the QLoRA Finetuning using the default parameters (`unsloth/Qwen2.5-1.5B-Instruct` model and `trl-lib/kto-mix-14k` dataset):

```bash
python main.py
```

### Inference

To run inference locally using the saved merged model:

```bash
python inference.py --model_name "hedrergudene/Qwen2.5-1.5B-KTO-Finetuning" --prompt "How do I bake a cake?"
```

Note: This requires a GPU and sufficient VRAM to load the model.
