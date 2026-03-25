"""Utility helpers for AVP encoding."""

import json
from typing import Any, Dict, Optional

import numpy as np


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Convert a numpy embedding to raw little-endian bytes."""
    arr = np.ascontiguousarray(embedding)
    # Force little-endian to match AVP wire format spec.
    # On LE systems (x86, most ARM) this is a no-op.
    if arr.dtype.byteorder not in ("<", "=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return arr.tobytes()


def embedding_to_json(embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bytes:
    """Serialize an embedding as JSON (for benchmark comparison)."""
    payload: Dict[str, Any] = {"embedding": embedding.tolist()}
    if metadata:
        payload["metadata"] = metadata
    return json.dumps(payload).encode("utf-8")
