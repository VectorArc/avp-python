"""ctypes wrappers for llama.cpp embedding weight extraction.

Provides GGUF weight dequantization for cross-model rosetta projection
on quantized models.

Requires: ``pip install avp[llamacpp]`` (installs llama-cpp-python, gguf)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import llama_cpp  # noqa: F401

    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False


def extract_gguf_embedding_weights(model_path: str) -> Any:
    """Extract and dequantize embed_tokens weight from a GGUF file.

    Reads the ``token_embd.weight`` tensor from the GGUF file and
    returns it as a float32 numpy array of shape ``[vocab_size, n_embd]``.

    Supports all quantization formats via the gguf package's built-in
    dequantization.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        numpy array [vocab_size, n_embd] in float32.
    """
    import numpy as np

    try:
        from gguf import GGUFReader
        from gguf.quants import dequantize as gguf_dequantize
    except ImportError:
        raise ImportError(
            "GGUF weight extraction requires the gguf package. "
            "Install with: pip install gguf"
        )

    reader = GGUFReader(model_path)

    # Find token_embd.weight by iterating tensors (get_tensor takes int index)
    tensor = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            tensor = t
            break

    if tensor is None:
        available = [t.name for t in reader.tensors[:10]]
        raise ValueError(
            f"token_embd.weight not found in {model_path}. "
            f"Available tensors (first 10): {available}"
        )

    # Dequantize: tensor.data is raw quantized bytes for Q4/Q6/Q8 etc.
    # gguf.quants.dequantize handles all quantization formats correctly.
    data = gguf_dequantize(tensor.data, tensor.tensor_type).astype(np.float32).flatten()

    # Infer shape from GGUF metadata (more reliable than tensor.shape
    # which may reflect quantized block layout, not logical dimensions).
    # Search all fields for embedding_length — model-agnostic, works
    # for any architecture (llama, qwen2, gemma, mistral, phi, etc.)
    n_embd_val = None
    for field_key in reader.fields:
        if field_key.endswith(".embedding_length"):
            field = reader.get_field(field_key)
            if field is not None:
                val = field.parts[-1]
                n_embd_val = int(val.item() if hasattr(val, "item") else val)
                break

    if n_embd_val is None:
        # Last resort: use tensor.shape
        shape = tuple(tensor.shape)
        if len(shape) == 2:
            n_embd_val = min(shape)  # smaller dim is n_embd
        else:
            raise ValueError(
                f"Cannot determine n_embd from {model_path}. "
                f"tensor.shape={tuple(tensor.shape)}, data.size={data.size}"
            )

    n_vocab_val = data.size // n_embd_val
    if n_vocab_val * n_embd_val != data.size:
        raise ValueError(
            f"Data size {data.size} not divisible by n_embd={n_embd_val}"
        )

    result = data.reshape(n_vocab_val, n_embd_val)

    # Sanity check: dequantized embedding norms should be reasonable
    mean_norm = float(np.linalg.norm(result, axis=1).mean())
    if mean_norm > 100:
        logger.warning(
            "Dequantized embedding norms look high (mean=%.1f). "
            "Expected ~0.5-2.0 for most models.",
            mean_norm,
        )

    return result
