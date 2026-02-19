"""Utility helpers for AVP encoding."""

import json
from typing import Any, Dict, Optional

import numpy as np


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Convert a numpy embedding to raw little-endian bytes."""
    arr = np.ascontiguousarray(embedding)
    return arr.tobytes()


def bytes_to_embedding(data: bytes, dtype: str, dim: int) -> np.ndarray:
    """Restore a numpy embedding from raw bytes."""
    dt = np.dtype(dtype).newbyteorder("<")
    arr = np.frombuffer(data, dtype=dt)
    if arr.shape[0] != dim:
        raise ValueError(f"Expected {dim} elements, got {arr.shape[0]}")
    return arr.copy()  # Return a writable copy


def embedding_to_json(embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bytes:
    """Serialize an embedding as JSON (for benchmark comparison)."""
    payload: Dict[str, Any] = {"embedding": embedding.tolist()}
    if metadata:
        payload["metadata"] = metadata
    return json.dumps(payload).encode("utf-8")
