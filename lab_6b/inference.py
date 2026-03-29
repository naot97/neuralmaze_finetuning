import os
import torch
import fire
from unsloth import FastLanguageModel
from snac import SNAC
from dotenv import load_dotenv

# Load environment variables (e.g. for HF_TOKEN to access private models)
load_dotenv()

# --- SPECIAL TOKEN IDS ---
TOKENISER_LENGTH = 128256
START_OF_HUMAN   = TOKENISER_LENGTH + 3
END_OF_HUMAN     = TOKENISER_LENGTH + 4
START_OF_AI      = TOKENISER_LENGTH + 5
END_OF_AI        = TOKENISER_LENGTH + 6
PAD_TOKEN        = TOKENISER_LENGTH + 7
END_OF_TEXT      = 128009
START_OF_SPEECH  = TOKENISER_LENGTH + 1
END_OF_SPEECH    = TOKENISER_LENGTH + 2


def redistribute_codes(code_list, snac_model):
    """Convert flat SNAC token list back into 3-layer codes and decode to audio."""
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))

    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0),
    ]
    audio_hat = snac_model.decode(codes)
    return audio_hat


def run_inference(
    model_name: str = "naot97/Orpheus-3B-TTS-Finetuning",
    prompt: str = "Hey there my name is Elise, and I'm a speech generation model that can sound like a person.",
    voice: str = None,
    output_path: str = "output.wav",
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    max_new_tokens: int = 1200,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
):
    """
    Run inference for the Orpheus-3B TTS model.

    Args:
        voice: Speaker name for multi-speaker models (e.g. "Elise"). None for single-speaker.
        output_path: Path to save the generated .wav file.
    """
    print(f"Loading model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    # Prepare text input with special tokens
    text_prompt = f"{voice}: {prompt}" if voice else prompt
    input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
    modified_input_ids = torch.cat(
        [start_token, input_ids, end_tokens], dim=1
    ).to("cuda")

    attention_mask = torch.ones_like(modified_input_ids).to("cuda")

    print("\n--- Generating audio tokens ---\n")

    generated_ids = model.generate(
        input_ids=modified_input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=1,
        eos_token_id=END_OF_SPEECH,
        use_cache=True,
    )

    # Extract audio tokens after START_OF_SPEECH (128257)
    token_to_find = START_OF_SPEECH
    token_to_remove = END_OF_SPEECH

    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        last_idx = token_indices[1][-1].item()
        cropped = generated_ids[:, last_idx + 1:]
    else:
        cropped = generated_ids

    # Remove end-of-speech tokens and trim to multiple of 7
    row = cropped[0]
    row = row[row != token_to_remove]
    new_length = (row.size(0) // 7) * 7
    code_list = [t - 128266 for t in row[:new_length].tolist()]

    if len(code_list) == 0:
        print("No audio tokens generated. Try a different prompt or increase max_new_tokens.")
        return

    # Decode with SNAC
    print("Decoding audio with SNAC codec...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    audio = redistribute_codes(code_list, snac_model)

    # Save to wav
    import torchaudio
    audio_data = audio.detach().squeeze().cpu()
    torchaudio.save(output_path, audio_data.unsqueeze(0), 24000)
    print(f"Audio saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(run_inference)
