# Lab 6a: QLoRA Vision Finetuning (Qwen3-VL-8B LaTeX OCR)

This lab finetunes Qwen3-VL-8B-Instruct for LaTeX OCR — converting images of handwritten math formulas into LaTeX code using QLoRA on both vision and language layers.

## Setup

### Using `pip`

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Using `Docker` (Recommended)

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  -e COMET_API_KEY="your-key" \
  -e HF_TOKEN="your-token" \
  theneuralmaze/finetuning-sessions bash
```

## Usage

### Training

```bash
python main.py
```

### Inference

Using a sample from the dataset:
```bash
python inference.py
```

Using a local image:
```bash
python inference.py --image_path "path/to/math_formula.png"
```

Note: This requires a GPU and sufficient VRAM to load the model.
