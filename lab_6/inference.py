import os
import torch
import fire
from unsloth import FastLanguageModel
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
from dotenv import load_dotenv

# Load environment variables (e.g. for HF_TOKEN to access private models)
load_dotenv()

# --- Qwen3 GRPO reasoning format ---
REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)


def build_chat_template():
    """Build the custom Qwen3 GRPO chat template with reasoning tags."""
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
        "{% else %}"
            "{{ '" + SYSTEM_PROMPT + "' + eos_token }}"
            "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '" + REASONING_START + "' }}"
        "{% endif %}"
    )
    return chat_template


def run_qwen3_inference(
    model_name: str = "naot97/Qwen3-0.6B-GRPO-Finetuning",
    prompt: str = "What is the sqrt of 101?",
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
):
    """
    Run inference for the Qwen3 GRPO reasoning model.
    """
    print(f"Loading model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    # Set custom chat template
    tokenizer.chat_template = build_chat_template()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    print("\n--- Model Response ---\n")

    # Stop on EOS or </SOLUTION>
    stop_ids = [tokenizer.eos_token_id]
    solution_end_ids = tokenizer.encode(SOLUTION_END, add_special_tokens=False)

    class StopOnSolutionEnd(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if len(solution_end_ids) == 0:
                return False
            generated = input_ids[0].tolist()
            if len(generated) >= len(solution_end_ids):
                return generated[-len(solution_end_ids):] == solution_end_ids
            return False

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids=input_ids,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_cache=True,
        eos_token_id=stop_ids,
        stopping_criteria=StoppingCriteriaList([StopOnSolutionEnd()]),
    )
    print("\n----------------------\n")


def run_gpt_oss_inference(
    model_name: str = "naot97/GPT-OSS-20B-GRPO-Finetuning",
    prompt: str = None,
    max_seq_length: int = 768,
    load_in_4bit: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    reasoning_effort: str = "low",
):
    """
    Run inference for the GPT-OSS-20B code optimization model.
    """
    if prompt is None:
        prompt = (
            "Create a new fast matrix multiplication function using only native Python code.\n"
            "You are given a list of list of numbers.\n"
            "Output your new function in backticks using the format below:\n"
            "```python\n"
            "def matmul(A, B):\n"
            "    return ...\n"
            "```"
        )

    print(f"Loading model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        reasoning_effort=reasoning_effort,
    ).to("cuda")

    print("\n--- Model Response ---\n")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids=input_ids,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_cache=True,
    )
    print("\n----------------------\n")


if __name__ == "__main__":
    fire.Fire({
        "qwen3": run_qwen3_inference,
        "gpt_oss": run_gpt_oss_inference,
    })
