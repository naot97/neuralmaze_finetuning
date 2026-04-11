# Lab 8: GLM-OCR Document Parsing

This lab demonstrates two ways to run **GLM-OCR** (Zhipu's open-source vision-language OCR model) for document understanding — text, tables, and formulas:

1. **Lightweight inference via Ollama** (`inference_ollama.py`) — wraps the model in a single API call. Good for quick OCR on a single image.
2. **Self-hosted SDK pipeline** (`inference_sdk.py`) — uses the `glmocr` SDK with PP-DocLayoutV3 layout detection, region cropping, and post-processed Markdown/JSON output. Suited for full document parsing.

## Files

- `inference_ollama.py` — sends a single image to a local Ollama server running the GLM-OCR model
- `inference_sdk.py` — runs the GLM-OCR SDK pipeline on a local image, configured by `config.yaml`
- `config.yaml` — full GLM-OCR SDK configuration (layout detection, OCR API, formatter)
- `GLM-Config` — Ollama Modelfile defining the optimized GLM-OCR model parameters
- `ntb.ipynb` — interactive notebook with both approaches
- `7cf7af6c-...jpg` — sample document image (Qwen3 Technical Report page)

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the Ollama model (for `inference.py`)

```bash
ollama create glm-ocr-optimized -f ./GLM-Config
```

This pulls the base `glm-ocr` model and applies the generation parameters defined in `GLM-Config`.

### 3. Self-hosted backend (for `main.py`)

The SDK expects an OCR backend (vLLM / SGLang / Ollama) reachable at the host/port specified in `config.yaml` under `pipeline.ocr_api`. Defaults: `localhost:11434` with `api_mode: ollama_generate`.

You also need PP-DocLayoutV3 weights at the path defined by `pipeline.layout.model_dir`.

## Usage

### Ollama inference (single image)

```bash
python inference_ollama.py --image "https://example.com/invoice.jpg"
python inference_ollama.py --image "local_doc.jpg" --prompt "Table Recognition:"
```

### SDK pipeline (full document parsing)

```bash
python inference_sdk.py
python inference_sdk.py --image_path "my_doc.jpg" --config_path "./config.yaml"
```

Available task prompts: `"Text recognition:"`, `"Table Recognition:"`, `"Formula Recognition:"`.
