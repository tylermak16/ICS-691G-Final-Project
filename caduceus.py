"""Load the pinned PH backbone and preserve the input nucleotide coordinates."""

import json
import os
from pathlib import Path

import torch

from dataset import file_digest


PROJECT = Path(__file__).resolve().parent
CONFIG = json.loads((PROJECT / "model.json").read_text(encoding="utf-8"))


def load_model():
    cache = Path(os.environ.get("SPLICE_MODEL_CACHE", "~/.local/share/splice-probing/hf")).expanduser()
    os.environ.setdefault("HF_MODULES_CACHE", str(cache / "modules"))
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The pinned backbone uses CUDA kernels. Run in the verified Linux CUDA environment.")
    files = {
        **CONFIG["code_sha256"], **CONFIG["config_sha256"],
        "model.safetensors": CONFIG["weights_sha256"],
    }
    snapshot = Path(snapshot_download(
        CONFIG["model_id"], revision=CONFIG["revision"], allow_patterns=list(files),
        cache_dir=cache / "hub",
    ))
    for filename, sha256 in files.items():
        if file_digest(snapshot / filename) != sha256:
            raise ValueError(f"Checkpoint file differs from the pinned SHA-256: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True, use_fast=False, local_files_only=True)
    model, info = AutoModel.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, output_loading_info=True,
    )
    unused = set(info["unexpected_keys"]) - {"lm_head.weight"}
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"] or unused:
        raise RuntimeError(f"Checkpoint backbone did not load cleanly: {info}")
    if model.config.rcps or not model.config.bidirectional or model.config.n_layer != 16 or model.config.d_model != 256:
        raise RuntimeError("Unexpected backbone architecture for the pinned PH checkpoint")
    return tokenizer, model.requires_grad_(False).eval().to("cuda"), info


def aligned_input_ids(tokenizer, sequences, motif_index):
    encoded = tokenizer(sequences, add_special_tokens=False, padding=False, truncation=False, return_tensors="pt")
    ids = encoded["input_ids"]
    expected = [tokenizer.convert_tokens_to_ids(list(sequence)) for sequence in sequences]
    if ids.tolist() != expected or (ids == tokenizer.unk_token_id).any():
        raise ValueError("Expected one known token per nucleotide and no padding or special tokens")
    if any(sequence[motif_index:motif_index + 2] != "GT" for sequence in sequences):
        raise ValueError(f"Batch does not contain GT at nucleotide index {motif_index}")
    g_id, t_id = tokenizer.convert_tokens_to_ids(["G", "T"])
    if not ((ids[:, motif_index] == g_id) & (ids[:, motif_index + 1] == t_id)).all():
        raise ValueError("GT token indices disagree with nucleotide indices")
    return ids.to("cuda")


def hidden_states(model, ids):
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("Representation extraction requires eval and inference mode")
    output = model(input_ids=ids, output_hidden_states=True, return_dict=True)
    if len(output.hidden_states) != model.config.n_layer + 1 or output.hidden_states[-1] is not output.last_hidden_state:
        raise RuntimeError("Checkpoint returned an unexpected set of hidden states")
    expected_shape = (*ids.shape, model.config.d_model)
    if any(state.shape != expected_shape or state.device != ids.device for state in output.hidden_states):
        raise RuntimeError(f"Expected device-aligned hidden states with shape {expected_shape}")
    return output.hidden_states


def gpu_memory():
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
