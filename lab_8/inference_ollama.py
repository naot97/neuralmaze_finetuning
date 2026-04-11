#!/usr/bin/env python3
"""
inference_ollama.py
--------------------------------------------------------------------------------
Lab 8: GLM-OCR via Ollama (Lightweight Inference)
--------------------------------------------------------------------------------
Author: The Neural Maze

DESCRIPTION:
Run GLM-OCR through a local Ollama server, without using the full SDK pipeline.
Image is downloaded (or loaded locally), resized to a max edge, base64-encoded,
then streamed to the Ollama model defined by `GLM-Config` (see Modelfile).

SETUP:
    ollama create glm-ocr-optimized -f ./GLM-Config

USAGE:
    python inference_ollama.py --image "https://example.com/doc.jpg"
    python inference_ollama.py --image "local_doc.jpg" --prompt "Table Recognition:"
"""

import base64
import time
from io import BytesIO

import fire
import ollama
import requests
from PIL import Image


def _load_image(source: str) -> Image.Image:
    """Load an image from a URL or local path."""
    if source.startswith(("http://", "https://")):
        print(f"Downloading image from {source}...")
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    print(f"Loading local image: {source}")
    return Image.open(source)


def _encode_image(img: Image.Image, max_dimension: int) -> str:
    """Resize and base64-encode an image as JPEG."""
    print(f"Original size: {img.size}")
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        print(f"Resized to: {img.size}")
    else:
        print("Image is already small enough, skipping resize.")

    buffered = BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def run_ocr(
    image: str = "https://marketplace.canva.com/EAE92Pl9bfg/6/0/1131w/canva-black-and-gray-minimal-freelancer-invoice-wPpAXSlmfF4.jpg",
    model: str = "glm-ocr-optimized",
    prompt: str = "Text recognition:",
    max_dimension: int = 1024,
):
    """
    Stream GLM-OCR output for an image via Ollama.

    Args:
        image: URL or local path of the image.
        model: Ollama model name (must be created from GLM-Config Modelfile).
        prompt: Task prompt — e.g. "Text recognition:", "Table Recognition:",
                "Formula Recognition:".
        max_dimension: Longest-edge cap for resizing (px).
    """
    img = _load_image(image)
    image_b64 = _encode_image(img, max_dimension)

    print(f"Sending to Ollama model '{model}' (waiting for first token)...")
    start_time = time.time()
    first_token = True

    stream = ollama.generate(
        model=model,
        prompt=prompt,
        images=[image_b64],
        stream=True,
    )

    for chunk in stream:
        if first_token:
            print(f"Time to first token: {time.time() - start_time:.2f}s\n")
            first_token = False
        print(chunk["response"], end="", flush=True)

    print(f"\n\nTotal processing time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    fire.Fire(run_ocr)
