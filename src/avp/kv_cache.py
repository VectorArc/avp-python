"""AVP KV-cache serialization and deserialization.

Handles conversion between HuggingFace KV-cache formats (legacy tuple and
DynamicCache) and flat byte representations for wire transfer.

Requires torch — this module uses lazy imports so the core SDK works without it.
"""

import struct
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .errors import DecodeError, ShapeMismatchError


def _require_torch():
    """Lazy import torch, raising a clear error if not available."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "torch is required for KV-cache operations. Install with: pip install avp[latent]"
        )


# KVCacheHeader binary format: num_layers(I) + num_kv_heads(I) + head_dim(I) + seq_len(I) + dtype(B)
_KV_HEADER_FMT = "<IIIIB"
_KV_HEADER_SIZE = struct.calcsize(_KV_HEADER_FMT)

# dtype byte encoding
_DTYPE_BYTE = {
    "float32": 0,
    "float16": 1,
    "bfloat16": 2,
}
_BYTE_DTYPE = {v: k for k, v in _DTYPE_BYTE.items()}


@dataclass
class KVCacheHeader:
    """Header describing the shape of a serialized KV-cache."""

    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    seq_len: int = 0
    dtype: str = "float16"

    def to_bytes(self) -> bytes:
        dtype_byte = _DTYPE_BYTE.get(self.dtype, 1)
        return struct.pack(
            _KV_HEADER_FMT,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.seq_len,
            dtype_byte,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "KVCacheHeader":
        if len(data) < _KV_HEADER_SIZE:
            raise DecodeError(
                f"KVCacheHeader too short: {len(data)} bytes, need {_KV_HEADER_SIZE}"
            )
        num_layers, num_kv_heads, head_dim, seq_len, dtype_byte = struct.unpack(
            _KV_HEADER_FMT, data[:_KV_HEADER_SIZE]
        )
        dtype = _BYTE_DTYPE.get(dtype_byte, "float16")
        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            dtype=dtype,
        )

    @property
    def header_size(self) -> int:
        return _KV_HEADER_SIZE


def serialize_kv_cache(past_key_values: Any) -> Tuple[bytes, KVCacheHeader]:
    """Serialize a HuggingFace KV-cache to bytes.

    Handles both legacy tuple format and DynamicCache.
    Layout: header + [K_l0][V_l0][K_l1][V_l1]... contiguous little-endian.

    Each K/V tensor has shape [batch=1, num_kv_heads, seq_len, head_dim].

    Args:
        past_key_values: HuggingFace past_key_values (tuple or DynamicCache).

    Returns:
        Tuple of (serialized bytes, KVCacheHeader).
    """
    torch = _require_torch()

    # Convert DynamicCache to legacy format if needed
    layers = dynamic_cache_to_legacy(past_key_values)

    if not layers:
        raise ValueError("Empty KV-cache")

    # Extract shape info from first layer
    k0, v0 = layers[0]
    # Shape: [batch, num_kv_heads, seq_len, head_dim]
    num_kv_heads = k0.shape[1]
    seq_len = k0.shape[2]
    head_dim = k0.shape[3]
    dtype_str = str(k0.dtype).replace("torch.", "")

    header = KVCacheHeader(
        num_layers=len(layers),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        dtype=dtype_str,
    )

    # Serialize all K/V tensors
    parts = [header.to_bytes()]
    for k, v in layers:
        # Squeeze batch dim, ensure contiguous
        k_flat = k.squeeze(0).contiguous().cpu()
        v_flat = v.squeeze(0).contiguous().cpu()
        parts.append(k_flat.numpy().tobytes())
        parts.append(v_flat.numpy().tobytes())

    return b"".join(parts), header


def deserialize_kv_cache(
    data: bytes,
    device: str = "cpu",
) -> Tuple[Tuple[Tuple[Any, Any], ...], KVCacheHeader]:
    """Deserialize bytes back to a KV-cache tuple.

    Args:
        data: Serialized KV-cache bytes (including header).
        device: Target device for tensors.

    Returns:
        Tuple of (past_key_values as legacy tuple, KVCacheHeader).
    """
    torch = _require_torch()
    import numpy as np

    header = KVCacheHeader.from_bytes(data)
    offset = _KV_HEADER_SIZE

    # Compute size per K or V tensor: num_kv_heads * seq_len * head_dim * dtype_size
    np_dtype = np.dtype(header.dtype)
    tensor_elements = header.num_kv_heads * header.seq_len * header.head_dim
    tensor_bytes = tensor_elements * np_dtype.itemsize

    torch_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(header.dtype, torch.float16)

    layers: List[Tuple[Any, Any]] = []
    for _ in range(header.num_layers):
        # Read K
        k_data = data[offset:offset + tensor_bytes]
        offset += tensor_bytes

        # Read V
        v_data = data[offset:offset + tensor_bytes]
        offset += tensor_bytes

        if len(k_data) < tensor_bytes or len(v_data) < tensor_bytes:
            raise DecodeError("KV-cache data truncated")

        k_np = np.frombuffer(k_data, dtype=np_dtype).reshape(
            header.num_kv_heads, header.seq_len, header.head_dim
        )
        v_np = np.frombuffer(v_data, dtype=np_dtype).reshape(
            header.num_kv_heads, header.seq_len, header.head_dim
        )

        k_tensor = torch.from_numpy(k_np.copy()).unsqueeze(0).to(device=device, dtype=torch_dtype)
        v_tensor = torch.from_numpy(v_np.copy()).unsqueeze(0).to(device=device, dtype=torch_dtype)

        layers.append((k_tensor, v_tensor))

    return tuple(layers), header


def dynamic_cache_to_legacy(past_key_values: Any) -> List[Tuple[Any, Any]]:
    """Convert DynamicCache or legacy tuple to a list of (K, V) tuples.

    Args:
        past_key_values: HuggingFace past_key_values.

    Returns:
        List of (key_tensor, value_tensor) per layer.
    """
    # Check for DynamicCache (transformers >= 4.36)
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_key_values, Cache):
            return list(past_key_values.to_legacy_cache())
    except ImportError:
        pass

    # Already a tuple/list of (K, V) pairs
    if isinstance(past_key_values, (tuple, list)):
        return [(layer[0], layer[1]) for layer in past_key_values]

    raise TypeError(f"Unsupported KV-cache type: {type(past_key_values).__name__}")


def legacy_to_dynamic_cache(
    past_key_values: Tuple[Tuple[Any, Any], ...],
) -> Any:
    """Convert legacy tuple KV-cache to DynamicCache.

    Args:
        past_key_values: Tuple of (K, V) per layer.

    Returns:
        DynamicCache instance.

    Raises:
        ImportError: If transformers is not available.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        raise ImportError(
            "transformers >= 4.36 required for DynamicCache. "
            "Install with: pip install avp[latent]"
        )

    return DynamicCache.from_legacy_cache(past_key_values)


def estimate_kv_cache_size(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: str = "float16",
) -> int:
    """Estimate KV-cache size in bytes for bandwidth planning.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
        seq_len: Sequence length (number of tokens).
        dtype: Data type string.

    Returns:
        Estimated size in bytes.
    """
    import numpy as np

    bytes_per_element = np.dtype(dtype).itemsize
    # 2 tensors (K+V) per layer, each [num_kv_heads, seq_len, head_dim]
    return 2 * num_layers * num_kv_heads * seq_len * head_dim * bytes_per_element
