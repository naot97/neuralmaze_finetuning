# Lab 1: Continued Pretraining (CPT) with Full Finetuning

This repository contains scripts for performing Continued Pretraining on small language models using Unsloth.

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

To start the Continued Pretraining using the default parameters (`Qwen/Qwen3-0.6B-Base` model and `Math-Pretraining-Data` dataset):

```bash
python main.py
```

### Inference

To run inference (requires `HF_ENDPOINT_URL` and `HF_API_TOKEN` environment variables):

```bash
python inference.py --prompt "Calculate 2+2" --model "Qwen3-0.6B-Base-CPT-Math"
```
