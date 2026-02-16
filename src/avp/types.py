"""AVP data types and dataclasses."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# --- Constants ---

MAGIC = b"\x41\x56"  # "AV"
PROTOCOL_VERSION = 0x01
HEADER_SIZE = 12  # 2 magic + 1 version + 1 flags + 4 payload_len + 4 metadata_len

CONTENT_TYPE = "application/avp+binary"
AVP_VERSION_HEADER = "1.0"


# --- Enums ---


class CompressionLevel(enum.Enum):
    """Zstd compression levels from the AVP spec."""

    NONE = 0
    FAST = 1
    BALANCED = 3
    MAX = 19


class DataType(enum.Enum):
    """Supported numpy dtypes for embeddings."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"


# --- Dataclasses ---


@dataclass
class AVPHeader:
    """Fixed-size header at the start of every AVP message."""

    magic: bytes = MAGIC
    version: int = PROTOCOL_VERSION
    flags: int = 0
    payload_length: int = 0
    metadata_length: int = 0

    @property
    def compressed(self) -> bool:
        return bool(self.flags & 0x01)

    @compressed.setter
    def compressed(self, value: bool) -> None:
        if value:
            self.flags |= 0x01
        else:
            self.flags &= ~0x01


@dataclass
class AVPMetadata:
    """Decoded metadata from an AVP message."""

    model_id: str = ""
    embedding_dim: int = 0
    data_type: str = "float32"
    compression: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class AVPMessage:
    """A fully decoded AVP message: header + metadata + embedding."""

    header: AVPHeader
    metadata: AVPMetadata
    embedding: np.ndarray
    raw_size: int = 0  # Total byte size of the encoded message
