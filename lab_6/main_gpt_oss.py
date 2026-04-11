#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "comet_ml",
#   "unsloth",
#   "transformers",
#   "datasets",
#   "trl",
#   "huggingface_hub",
#   "python-dotenv",
#   "numpy",
# ]
# ///

"""
main_gpt_oss.py
--------------------------------------------------------------------------------
Hugging Face Job: GRPO Finetuning Experiment (GPT-OSS-20B Code Optimization)
--------------------------------------------------------------------------------
Author: The Neural Maze
Context: Lecture on LLM Finetuning Techniques.

DESCRIPTION:
This script trains GPT-OSS-20B via GRPO to generate fast pure-Python matrix
multiplication code. The model learns to write correct, stdlib-only matmul
implementations and is rewarded for both correctness and speed (benchmarked
against numpy.matmul).

KEY CONCEPTS DEMONSTRATED:
1. QLoRA finetuning (4-bit) with embedding offloading for a 20B model.
2. GRPO with code-execution reward functions: sandboxed eval, stdlib-only
   enforcement, correctness checking, and speed benchmarking.
3. Synthetic dataset: a single prompt repeated N times — the diversity comes
   from the model's own generations during GRPO.

CLI INSTRUCTION
hf jobs uv run main_gpt_oss.py `
    --flavor a10g-small `
    --timeout 2h `
    --max_steps 100 `
    -e COMET_PROJECT_NAME="finetuning-sessions-lab5b-gptoss" `
    -s COMET_API_KEY="XXXX" `
    -s HF_TOKEN="XXXX"
"""

import sys
import os
import ast
import gc
import re
import time
import types
import signal
import statistics
import sysconfig
from pathlib import Path
from contextlib import contextmanager
import logging as log

import comet_ml
import torch
import fire
import numpy as np
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv

# Load environment variables (e.g. for HF_TOKEN and COMET_API_KEY)
load_dotenv()

# --- LOGGING SETUP ---
root = log.getLogger()
root.setLevel(log.INFO)
handler = log.StreamHandler(sys.stdout)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# --- PROMPT ---
MATMUL_PROMPT = """
Create a new fast matrix multiplication function using only native Python code.
You are given a list of list of numbers.
Output your new function in backticks using the format below:
```python
def matmul(A, B):
    return ...
```
""".strip()


# =============================================================================
# Helper utilities
# =============================================================================

def generate_random_matrices(seed=3407, n=256):
    """Generate two random matrices suitable for multiplication."""
    random_state = np.random.RandomState(seed)
    n, k, m = random_state.randint(1, n + 1, size=3)
    A = np.random.uniform(-10, 10, size=(n, k))
    B = np.random.uniform(-10, 10, size=(k, m))
    return A, A.tolist(), B, B.tolist()


def calculate_difference(pred, real):
    """Return (amax_error, mse_error) between prediction and real result."""
    if pred is None:
        return 5, 5
    try:
        difference = pred - real
    except:
        return 5, 5
    amax_error = float(np.amax(difference))
    mse_error = float(np.mean(np.square(difference)))
    return amax_error, mse_error


def extract_function(text):
    """Extract a `def matmul(A, B):` function from markdown code blocks."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def matmul(A, B):"):
            return fx
    return None


# --- stdlib import checker ---

def _stdlib_names():
    """Build a set of canonical stdlib top-level module/package names."""
    names = {m.lower() for m in getattr(sys, "stdlib_module_names", set())}
    names |= {m.lower() for m in sys.builtin_module_names}
    names.add("__future__")
    try:
        stdlib_dir = Path(sysconfig.get_path("stdlib"))
        if stdlib_dir.exists():
            for p in stdlib_dir.iterdir():
                if p.name == "site-packages":
                    continue
                if p.suffix == ".py":
                    names.add(p.stem.lower())
                elif p.is_dir() and (p / "__init__.py").exists():
                    names.add(p.name.lower())
    except Exception:
        pass
    return names


_STDLIB_SET = _stdlib_names()


def check_only_stdlib_imports(code):
    """
    Return (ok, info) — ok is True if all imports are stdlib-only.
    info contains stdlib, non_stdlib lists and relative_imports count.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "error": f"SyntaxError: {e}",
            "stdlib": [],
            "non_stdlib": [],
            "relative_imports": 0,
        }

    abs_imports = set()
    relative_count = 0

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                abs_imports.add(alias.name.split(".")[0])

        def visit_ImportFrom(self, node):
            nonlocal relative_count
            if (node.level or 0) > 0:
                relative_count += 1
            else:
                if node.module:
                    abs_imports.add(node.module.split(".")[0])

    Visitor().visit(tree)

    stdlib_found = sorted(m for m in abs_imports if m.lower() in _STDLIB_SET)
    non_stdlib = sorted(m for m in abs_imports if m.lower() not in _STDLIB_SET)

    return len(non_stdlib) == 0, {
        "stdlib": stdlib_found,
        "non_stdlib": non_stdlib,
        "relative_imports": relative_count,
    }


def create_locked_down_function(function_code):
    """Exec function code in an empty namespace and strip globals for sandboxing."""
    output_function = {}
    exec(function_code, {}, output_function)
    new_matmul = output_function["matmul"]
    new_matmul = types.FunctionType(new_matmul.__code__, {})
    return new_matmul


# --- Benchmarker ---

class _TimeoutError(Exception):
    pass


@contextmanager
def _time_limit(seconds):
    def _handler(signum, frame):
        raise _TimeoutError(f"Timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


class Benchmarker:
    def __init__(self, trials=3, loops=1, timeout=30):
        self.buffer = np.zeros(2 * 1024 * 1024 * 1024, dtype=np.uint8)
        self.trials = trials
        self.loops = loops
        assert timeout > 0
        self.timeout = timeout

    def thrash(self):
        self.buffer ^= 1
        return int(self.buffer[::4096].sum())

    def benchmark(self, function, arguments):
        assert len(arguments) == self.loops
        samples = []
        exceptions = []
        timed_out = 0
        for _ in range(self.trials):
            gc.collect()
            gc.disable()
            self.thrash()
            t_start = time.perf_counter_ns()
            for i in range(self.loops):
                try:
                    with _time_limit(self.timeout):
                        function(*arguments[i])
                except _TimeoutError:
                    timed_out += 1
                except Exception as e:
                    exceptions.append(str(e))
            t_end = time.perf_counter_ns()
            gc.enable()
            samples.append((t_end - t_start) // max(1, self.loops))
        return {
            "median_ns": int(statistics.median(samples)),
            "mean_ns": int(statistics.fmean(samples)),
            "stdev_ns": int(statistics.pstdev(samples) if len(samples) > 1 else 0),
            "exceptions": exceptions,
            "timeouts": timed_out,
        }


# =============================================================================
# Reward functions
# =============================================================================

def build_reward_functions(benchmarker):
    """Build the 4 reward functions for GRPO code-optimization training."""

    def function_works(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                ok, info = check_only_stdlib_imports(function)
            if function is None or "error" in info:
                score = -2.0
            else:
                try:
                    create_locked_down_function(function)
                    score = 1.0
                except:
                    score = -0.5
            scores.append(score)
        return scores

    def no_cheating(completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                ok, info = check_only_stdlib_imports(function)
            else:
                ok = False
            scores.append(1.0 if ok else -20.0)
        return scores

    def correctness_check(completions, **kwargs):
        scores = []
        A, A_list, B, B_list = generate_random_matrices(
            seed=np.random.randint(10000), n=128
        )
        for completion in completions:
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                ok, info = check_only_stdlib_imports(function)
            if function is None or "error" in info:
                scores.append(0)
                continue
            try:
                new_matmul = create_locked_down_function(function)
            except:
                scores.append(0)
                continue
            try:
                pred = new_matmul(A_list.copy(), B_list.copy())
            except:
                scores.append(-2.0)
                continue

            true = np.matmul(A, B)
            amax_error, mse_error = calculate_difference(pred, true)

            machine_epsilon = 100 * np.finfo(np.float64).eps
            score = 0
            if   amax_error >= 3:                    score = -3.0
            elif amax_error >= 2:                    score = -2.5
            elif amax_error >= 1:                    score = -2.0
            elif amax_error >= 0.5:                  score = -1.0
            elif amax_error >= 100 * machine_epsilon: score = 0.0
            elif amax_error >= machine_epsilon:       score = 1.0
            else:                                     score = 3.0

            if   mse_error >= 3:                    score += -3.0
            elif mse_error >= 2:                    score += -2.5
            elif mse_error >= 1:                    score += -2.0
            elif mse_error >= 0.5:                  score += -1.0
            elif mse_error >= 100 * machine_epsilon: score += 0.0
            elif mse_error >= machine_epsilon:       score += 1.0
            else:                                     score += 3.0
            scores.append(score)
        return scores

    def speed_check(completions, **kwargs):
        scores = []
        A, A_list, B, B_list = generate_random_matrices(
            seed=np.random.randint(10000), n=256
        )
        numpy_results = benchmarker.benchmark(np.matmul, [(A, B)])
        for completion in completions:
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                ok, info = check_only_stdlib_imports(function)
            if function is None or "error" in info:
                scores.append(0)
                continue
            try:
                new_matmul = create_locked_down_function(function)
            except:
                scores.append(0)
                continue

            new_results = benchmarker.benchmark(
                new_matmul, [(A_list.copy(), B_list.copy())]
            )

            negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
            positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
            score = negative if new_results["median_ns"] >= numpy_results["median_ns"] else positive
            score = max(-10, min(10, score))
            scores.append(score)

        gc.collect()
        torch.cuda.empty_cache()
        return scores

    return [function_works, no_cheating, correctness_check, speed_check]


# =============================================================================
# Main
# =============================================================================

def main(
    # --- MODEL PARAMETERS ---
    model_name: str = "unsloth/gpt-oss-20b",
    load_in_4bit: bool = True,
    offload_embedding: bool = True,
    max_seq_length: int = 768,

    # --- LORA PARAMETERS ---
    lora_r: int = 4,
    lora_alpha: int = 8,

    # --- DATASET PARAMETERS ---
    dataset_size: int = 1000,
    reasoning_effort: str = "low",

    # --- GRPO TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "naot97/GPT-OSS-20B-GRPO-Finetuning",
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-5,
    max_steps: int = 100,
    num_generations: int = 2,
    benchmark_trials: int = 3,
    benchmark_timeout: int = 10,
    report_to: str = "comet_ml",
) -> None:

    log.info("================================================================")
    log.info("   STARTING GRPO FINETUNING EXPERIMENT (GPT-OSS-20B Code Opt)  ")
    log.info("================================================================")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD MODEL & ADD LORA ADAPTERS (QLoRA)
    # --------------------------------------------------------------------------
    log.info(f"Loading Base Model in 4-bit (QLoRA): {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        offload_embedding=offload_embedding,
    )

    log.info("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    log.info("Model prepared for QLoRA training.")

    # --------------------------------------------------------------------------
    # STEP 2: PREPARE SYNTHETIC DATASET
    # --------------------------------------------------------------------------
    log.info(f"Creating synthetic dataset ({dataset_size} copies of matmul prompt)...")

    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": MATMUL_PROMPT}],
            "answer": 0,
            "reasoning_effort": reasoning_effort,
        }
    ] * dataset_size)

    maximum_length = len(tokenizer(MATMUL_PROMPT)["input_ids"])
    log.info(f"Prompt token length: {maximum_length}")

    # --------------------------------------------------------------------------
    # STEP 3: BUILD REWARD FUNCTIONS & CONFIGURE GRPO
    # --------------------------------------------------------------------------
    log.info("Initializing benchmarker and reward functions...")
    benchmarker = Benchmarker(
        trials=benchmark_trials, timeout=benchmark_timeout
    )
    reward_funcs = build_reward_functions(benchmarker)

    max_prompt_length = maximum_length + 1
    max_completion_length = max_seq_length - max_prompt_length

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=max_steps,
        report_to=[report_to],
        output_dir=output_dir,
        seed=3407,
    )

    # --------------------------------------------------------------------------
    # STEP 4: EXECUTE GRPO TRAINING
    # --------------------------------------------------------------------------
    log.info("Starting GRPO Training Loop...")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = torch.cuda.memory_allocated() / 1024**3

    log.info(f"GPU: {gpu_stats.name}")
    log.info(f"Total VRAM Available: {gpu_stats.total_memory / 1024**3:.2f} GB")
    log.info(f"Pre-Train VRAM Used: {start_gpu_memory:.2f} GB")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer_stats = trainer.train()

    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3

    log.info(f"GRPO Training Complete. Final Loss: {trainer_stats.training_loss:.4f}")
    log.info(f"Peak VRAM Used: {peak_gpu_memory:.2f} GB")

    # --------------------------------------------------------------------------
    # STEP 5: SAVE & PUSH FULL MODEL (MERGED)
    # --------------------------------------------------------------------------
    log.info(f"Merging and pushing FULL model to Hugging Face Hub: {hub_model_id}...")

    model.push_to_hub_merged(
        hub_model_id,
        tokenizer,
        save_method="merged_16bit",
    )

    log.info("Push complete!")


if __name__ == "__main__":
    fire.Fire(main)
