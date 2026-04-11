# Lab 6b: LoRA Finetuning (Orpheus-3B TTS)

This lab finetunes the Orpheus-3B text-to-speech model using LoRA. Orpheus treats TTS as a language modeling task — text is tokenized normally, and audio is encoded into discrete SNAC codec tokens.

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

Generate speech and save to wav:
```bash
python inference.py --prompt "Hello, this is a test of the speech model."
```

With a specific speaker voice (multi-speaker model):
```bash
python inference.py --prompt "Hello world" --voice "Elise" --output_path "elise.wav"
```

Note: This requires a GPU and sufficient VRAM to load the model.
