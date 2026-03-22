"""ctypes wrappers for llama.cpp hidden state extraction and embedding injection.

Exposes two capabilities not available in llama-cpp-python's high-level API:

1. **cb_eval callback** — intercepts per-layer hidden states during forward pass.
   Set on ``llama_context_params.cb_eval`` before context creation.

2. **batch.embd injection** — sets the ``embd`` pointer on a ``llama_batch``
   to inject pre-computed embeddings instead of token IDs.

These use the same C API functions that llama.cpp's own tools use
(cvector-generator, activation-capture, multimodal models).

Requires: ``pip install llama-cpp-python>=0.3``
"""

import ctypes
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import llama_cpp  # noqa: F401

    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False


# ggml_backend_sched_eval_callback type:
# bool (*)(struct ggml_tensor * t, bool ask, void * user_data)
if HAS_LLAMACPP:
    EVAL_CALLBACK_TYPE = ctypes.CFUNCTYPE(
        ctypes.c_bool,           # return: bool
        ctypes.c_void_p,         # t: ggml_tensor*
        ctypes.c_bool,           # ask: bool
        ctypes.c_void_p,         # user_data: void*
    )
else:
    EVAL_CALLBACK_TYPE = None


# ggml_tensor field offsets (from ggml.h)
# struct ggml_tensor {
#   enum ggml_type type;        // offset 0, 4 bytes
#   void* buffer;               // offset 8 (aligned), 8 bytes
#   int64_t ne[4];              // offset 16, 32 bytes
#   size_t nb[4];               // offset 48, 32 bytes
#   enum ggml_op op;            // offset 80, 4 bytes
#   int32_t op_params[16];      // offset 84, 64 bytes
#   int32_t flags;              // offset 148, 4 bytes
#   void* src[10];              // offset 152 (aligned to 8), 80 bytes
#   void* view_src;             // offset 232, 8 bytes
#   size_t view_offs;           // offset 240, 8 bytes
#   void* data;                 // offset 248, 8 bytes
#   char name[64];              // offset 256, 64 bytes
# }
_TENSOR_NE_OFFSET = 16       # int64_t ne[4]
_TENSOR_DATA_OFFSET = 248    # void* data
_TENSOR_NAME_OFFSET = 256    # char name[64]
_TENSOR_TYPE_OFFSET = 0      # enum ggml_type (int32)

# ggml_type values
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def get_tensor_name(tensor_ptr: int) -> str:
    """Extract the name string from a ggml_tensor pointer."""
    name_ptr = tensor_ptr + _TENSOR_NAME_OFFSET
    return ctypes.string_at(name_ptr, 64).split(b"\x00")[0].decode("ascii", errors="replace")


def get_tensor_shape(tensor_ptr: int) -> tuple:
    """Extract ne[0..3] dimensions from a ggml_tensor pointer."""
    ne_ptr = tensor_ptr + _TENSOR_NE_OFFSET
    ne = (ctypes.c_int64 * 4).from_address(ne_ptr)
    return (ne[0], ne[1], ne[2], ne[3])


def get_tensor_data_ptr(tensor_ptr: int) -> int:
    """Get the raw data pointer from a ggml_tensor."""
    data_field = ctypes.c_void_p.from_address(tensor_ptr + _TENSOR_DATA_OFFSET)
    return data_field.value or 0


def get_tensor_type(tensor_ptr: int) -> int:
    """Get the ggml_type enum value from a ggml_tensor."""
    type_field = ctypes.c_int32.from_address(tensor_ptr + _TENSOR_TYPE_OFFSET)
    return type_field.value


def copy_tensor_data_f32(tensor_ptr: int, n_elements: int) -> Any:
    """Copy tensor data to a numpy array (float32)."""
    import numpy as np

    data_ptr = get_tensor_data_ptr(tensor_ptr)
    if data_ptr == 0:
        return None
    arr = (ctypes.c_float * n_elements).from_address(data_ptr)
    return np.array(arr, dtype=np.float32, copy=True)


def make_eval_callback(
    target_layer: int,
    capture_dict: dict,
) -> Any:
    """Create a cb_eval callback that captures a specific layer's output.

    Args:
        target_layer: Layer index to capture (e.g., -1 for last layer,
            determined at runtime from the tensor name ``l_out-{N}``).
        capture_dict: Mutable dict where captured data is stored.
            Keys: ``"data"`` (numpy array), ``"shape"`` (tuple),
            ``"layer"`` (int).

    Returns:
        A ctypes callback function suitable for ``llama_context_params.cb_eval``.
        Keep a reference to prevent garbage collection.
    """
    if EVAL_CALLBACK_TYPE is None:
        raise ImportError("llama-cpp-python required for eval callback")

    target_name = f"l_out-{target_layer}" if target_layer >= 0 else None

    @EVAL_CALLBACK_TYPE
    def callback(tensor_ptr, ask, user_data):
        try:
            # Check active flag — disabled during generation to avoid
            # interfering with normal forward pass (data sync overhead)
            if not capture_dict.get("active", True):
                return False

            name = get_tensor_name(tensor_ptr)

            if ask:
                # During ask phase: return True for tensors we want to observe
                if target_name is not None:
                    return name == target_name
                # If target_layer == -1, capture all l_out-* to find the last one
                return name.startswith("l_out-")

            # During eval phase: copy the tensor data
            if not name.startswith("l_out-"):
                return False

            shape = get_tensor_shape(tensor_ptr)
            n_elements = shape[0] * max(shape[1], 1)
            data = copy_tensor_data_f32(tensor_ptr, n_elements)

            if data is not None:
                layer_idx = int(name.split("-")[1])
                # For target_layer == -1, keep updating (last one wins)
                if target_layer < 0 or layer_idx == target_layer:
                    capture_dict["data"] = data.reshape(max(shape[1], 1), shape[0])
                    capture_dict["shape"] = (max(shape[1], 1), shape[0])
                    capture_dict["layer"] = layer_idx

        except Exception:
            pass  # Never crash the inference loop

        return False

    return callback


# llama_batch struct layout (from llama.h):
# struct llama_batch {
#   int32_t n_tokens;     // offset 0
#   // padding             // offset 4 (4 bytes for alignment)
#   llama_token* token;   // offset 8
#   float* embd;          // offset 16
#   llama_pos* pos;       // offset 24
#   ...
# }
_BATCH_EMBD_OFFSET = 16  # float* embd (after n_tokens + padding + token ptr)


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


def set_batch_embeddings(batch_ptr: int, embeddings_ptr: int) -> None:
    """Set the embd pointer on a llama_batch struct.

    This enables embedding injection (inputs_embeds equivalent).
    The token pointer should be NULL when embd is set.

    Args:
        batch_ptr: Address of the llama_batch struct.
        embeddings_ptr: Address of the float array [n_tokens * n_embd].
    """
    embd_field = ctypes.c_void_p.from_address(batch_ptr + _BATCH_EMBD_OFFSET)
    embd_field.value = embeddings_ptr
