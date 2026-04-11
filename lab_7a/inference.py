import os
import torch
import fire
from unsloth import FastVisionModel
from transformers import TextStreamer
from dotenv import load_dotenv

# Load environment variables (e.g. for HF_TOKEN to access private models)
load_dotenv()


def run_inference(
    model_name: str = "naot97/Qwen3-VL-2B-LaTeX-OCR-Finetuning",
    image_path: str = None,
    instruction: str = "Write the LaTeX representation for this image.",
    load_in_4bit: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 1.5,
    min_p: float = 0.1,
):
    """
    Run inference for the Qwen3-VL vision model (LaTeX OCR).

    Args:
        image_path: Path to a local image file. If None, uses the first
                    sample from the unsloth/LaTeX_OCR dataset.
    """
    print(f"Loading model: {model_name}...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )

    FastVisionModel.for_inference(model)

    # Load image
    if image_path is not None:
        from PIL import Image
        image = Image.open(image_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
        image = dataset[0]["image"]
        print(f"Using sample image from dataset (index 0)")

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}
    ]

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    print("\n--- Model Response ---\n")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
    )
    print("\n----------------------\n")


if __name__ == "__main__":
    fire.Fire(run_inference)
