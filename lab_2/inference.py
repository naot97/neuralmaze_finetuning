import os
import torch
import fire
from unsloth import FastLanguageModel
from transformers import TextStreamer
from dotenv import load_dotenv

# Load environment variables (e.g. for HF_TOKEN to access private models)
load_dotenv()

def run_local_inference(
    model_name: str = "Qwen3-0.6B-Full-Finetuning",
    prompt: str = "Hello, how can I help you today?",
    max_seq_length: int = 2048,
    load_in_4bit: bool = False, # Set to True for larger models if VRAM is limited
):
    """
    Run inference locally using Unsloth.
    """
    print(f"Loading model: {model_name}...")
    
    # 1. Load Model and Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None, # Auto-detect
        load_in_4bit = load_in_4bit,
    )
    
    # FastLanguageModel for inference
    FastLanguageModel.for_inference(model) 

    # 2. Format the prompt using the model's chat template
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    # 3. Execution with Streaming
    print("\n--- Model Response ---\n")
    
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids = input_ids,
        streamer = text_streamer,
        max_new_tokens = 1000,
        use_cache = True,
    )
    print("\n----------------------\n")

if __name__ == "__main__":
    fire.Fire(run_local_inference)
