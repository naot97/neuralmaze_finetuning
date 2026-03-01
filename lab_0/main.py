#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "opik>=1.9.100",
#   "unsloth"
# ]
# ///


"""
main.py
Minimal python script to illustrate how huggingface jobs are run
using uv scripts (Python scripts with inline dependencies).
hf jobs uv run --flavor a10g-small main.py --input_text "'The answer is 42'"
"""

# Dependencies
import sys
import logging as log
import torch
import unsloth
import fire

# Custom logs handling
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Main method   
def main(input_text:str) -> None:

    log.info("--- Environment Diagnostics ---")

    # 1. CUDA Version
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
    else:
        cuda_ver = "N/A (CPU Mode)"
        gpu_name = "None"
        gpu_count = 0

    # 2. Unsloth Version
    unsloth_ver = getattr(unsloth, "__version__", "Unknown")

    # 3. Miscellaneous
    log.info(f"PyTorch Version:  {torch.__version__}")
    log.info(f"CUDA Version:     {cuda_ver}")
    log.info(f"Unsloth Version:  {unsloth_ver}")
    log.info(f"GPUs Available:   {gpu_count} ({gpu_name})")

    # 4. Sample input
    if not isinstance(input_text, str):
        log.error(f"Input parameter `input_text` is not a string.")
        raise ValueError(f"Input parameter `input_text` is not a string.")
    log.info(f"Input received:   {input_text}")
    log.info("--- Diagnostics Complete ---")

# Where the code is actually run
if __name__ == "__main__":
    fire.Fire(main)