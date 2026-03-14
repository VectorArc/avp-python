"""Model loading, device selection, and seed utilities for benchmarks."""

import os
import random
import time
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str]) -> str:
    """Auto-detect best available device."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str, attn_implementation: Optional[str] = None):
    """Load model and tokenizer, return (model, tokenizer, connector, identity).

    Args:
        attn_implementation: Override attention implementation (e.g. "eager" for
            output_attentions support — SDPA silently ignores it).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from avp.connectors.huggingface import HuggingFaceConnector

    print(f"Loading model: {model_name} on {device}...")
    t0 = time.perf_counter()

    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    kwargs = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
        print(f"  Using attention implementation: {attn_implementation}")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    connector = HuggingFaceConnector(model=model, tokenizer=tokenizer, device=device)
    identity = connector.get_model_identity()

    elapsed = time.perf_counter() - t0
    print(f"Model loaded in {elapsed:.1f}s. Identity: {identity.model_family}, "
          f"hidden_dim={identity.hidden_dim}, layers={identity.num_layers}, "
          f"kv_heads={identity.num_kv_heads}")

    return model, tokenizer, connector, identity
