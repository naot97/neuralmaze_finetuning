# Lab 0: Environment Diagnostics

This repository contains a minimal script to illustrate how to run environment diagnostics for Unsloth and CUDA.

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
  theneuralmaze/finetuning-sessions bash
```

To run the entrypoint script (`main.py`) directly (you must provide an input):
```bash
docker run --rm --gpus all \
  -v $(pwd):/app \
  theneuralmaze/finetuning-sessions python3 main.py --input_text "Hello World"
```

## Usage

### Environment Diagnostics

To run the environment diagnostics script:

```bash
python main.py --input_text "Environment Test"
```

### Inference Sample

To run the inference sample (requires `HF_ENDPOINT_URL` and `HF_API_TOKEN` environment variables):

```bash
python inference_sample.py --prompt "Calculate 2+2" --model "Qwen3-0.6B-Base-CPT-Math"
```
