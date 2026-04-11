#!/usr/bin/env python3
"""
inference_sdk.py
--------------------------------------------------------------------------------
Lab 8: GLM-OCR Document Parsing (Self-hosted SDK Pipeline)
--------------------------------------------------------------------------------
Author: The Neural Maze

DESCRIPTION:
Parse a document image (or PDF) using the GLM-OCR SDK in self-hosted mode.
The SDK runs PP-DocLayoutV3 for layout detection, then sends each region to a
self-hosted GLM-OCR vLLM/SGLang/Ollama backend for text/table/formula
recognition. Configuration is read from `config.yaml`.

USAGE:
    python inference_sdk.py
    python inference_sdk.py --image_path "my_doc.jpg" --config_path "./config.yaml"
"""

import fire
from PIL import Image
from glmocr import GlmOcr


def run_ocr(
    image_path: str = "7cf7af6c-0581-4fdc-a20f-7123aab8c0a2_3308x2339.jpg",
    config_path: str = "./config.yaml",
    max_dimension: int = 1024,
    resize: bool = True,
):
    """
    Run the GLM-OCR SDK pipeline on a local document image.

    Args:
        image_path: Path to the input image (or PDF).
        config_path: Path to the GLM-OCR config YAML.
        max_dimension: Resize the longest edge to this value (px) before OCR.
        resize: Whether to downscale the image to speed up CPU inference.
    """
    if resize:
        with Image.open(image_path) as img:
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                img.save(image_path)
                print(f"Resized {image_path} to {img.size} for faster inference.")

    print("Initializing GLM-OCR SDK...")
    with GlmOcr(config_path=config_path) as parser:
        print("Analyzing document structure...")
        result = parser.parse(image_path)

        print("\n" + "=" * 20 + " OCR RESULT (Markdown) " + "=" * 20)
        print(result.markdown_result)
        print("=" * 63)

        print("\n" + "=" * 20 + " OCR RESULT (JSON) " + "=" * 20)
        print(result.json_result)
        print("=" * 59)


if __name__ == "__main__":
    fire.Fire(run_ocr)
