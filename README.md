# Neural Maze — Finetuning Sessions

Hands-on labs for the **[Finetuning Sessions](https://theneuralmaze.substack.com/p/finetuning-sessions-course-overview)** course by [The Neural Maze](https://theneuralmaze.substack.com/) (Miguel Otero Pedrido & Antonio Zarauz Moreno) — a guided journey from pre-training to reasoning models, covering LoRA, QLoRA, RLHF, GRPO, multimodal finetuning, and deployment.

Each `lab_N/` directory is a self-contained project with its own `main.py`, `inference.py`, `pyproject.toml`, and `README.md`. Training scripts use **Unsloth** + **TRL** and can run locally, via Docker, or remotely on **Hugging Face Jobs**.

## Course Map

| Lab | Lesson | Topic | Technique |
|---|---|---|---|
| [`lab_0`](lab_0/) | Lesson 0 | Environment diagnostics — verify CUDA, Unsloth, GPU, and HF endpoint setup | — |
| [`lab_1`](lab_1/) | Lesson 1 | Continued Pretraining (CPT) on raw text — math domain adaptation | Full finetuning |
| [`lab_2`](lab_2/) | Lesson 2 | Supervised Finetuning on instruction/response pairs | Full SFT |
| [`lab_3`](lab_3/) | Lesson 3 | LoRA — parameter-efficient finetuning with low-rank adapters | LoRA SFT |
| [`lab_4`](lab_4/) | Lesson 4 | QLoRA — 4-bit NF4 quantized base + LoRA adapters | QLoRA SFT |
| [`lab_5`](lab_5/) | Lesson 5 | Preference optimization (alignment beyond SFT) — KTO | QLoRA + KTO |
| [`lab_6`](lab_6/) | Lesson 6 | GRPO — reasoning models (Qwen3 math) & code optimization (GPT-OSS-20B) | GRPO |
| [`lab_7a`](lab_7a/) | Lesson 7 | Vision finetuning — Qwen3-VL LaTeX OCR | QLoRA Vision |
| [`lab_7b`](lab_7b/) | Lesson 7 | Speech / TTS — Orpheus-3B with SNAC codec | LoRA TTS |
| [`lab_8`](lab_8/) | Lesson 8 | Deployment — GLM-OCR document parsing via Ollama & self-hosted SDK | Inference only |

## Running a Lab

### Local (requires NVIDIA GPU + CUDA)

```bash
cd lab_N
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py                                         # train with defaults
python main.py --model_name "..." --learning_rate 1e-5  # override
python inference.py --model_name "..." --prompt "..."  # after training
```

The shared dependencies live in [`requirements-common.txt`](requirements-common.txt) at the repo root; each lab's `requirements.txt` references it via `-r ../requirements-common.txt` plus any lab-specific extras.

### Docker (recommended — handles CUDA + Unsloth)

```bash
cd lab_N
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  -e COMET_API_KEY="..." \
  -e HF_TOKEN="..." \
  theneuralmaze/finetuning-sessions bash
```

### Hugging Face Jobs (remote execution)

Each `main.py` declares its own PEP 723 `# /// script` block so it can be uploaded as a standalone script:

```bash
cd lab_N
hf jobs uv run main.py --flavor a10g-small
```

## Environment Variables

| Variable | Used for |
|---|---|
| `HF_TOKEN` | Hugging Face auth (pulling models, pushing to the Hub) |
| `COMET_API_KEY` | [Comet ML](https://www.comet.com/) experiment tracking (`report_to=["comet_ml"]` in training args) |
| `COMET_PROJECT_NAME` | Comet project namespace for the lab |
| `HF_ENDPOINT_URL` / `HF_API_TOKEN` | Remote inference against deployed HF Endpoints (`lab_0`, `lab_1`) |

Labs load them automatically via `python-dotenv` — drop a `.env` file inside the lab directory.

## Credits

- **Course & curriculum**: Miguel Otero Pedrido & Antonio Zarauz Moreno — [The Neural Maze](https://theneuralmaze.substack.com/)
- **Docker image**: `theneuralmaze/finetuning-sessions`
- **Core stack**: [Unsloth](https://github.com/unslothai/unsloth), [Hugging Face TRL](https://github.com/huggingface/trl), [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets)
