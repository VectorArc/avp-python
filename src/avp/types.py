"""AVP data types and dataclasses."""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# --- Constants ---

MAGIC = b"\x41\x56"  # "AV"
PROTOCOL_VERSION = 0x01
HEADER_SIZE = 12  # 2 magic + 1 version + 1 flags + 4 payload_len + 4 metadata_len

CONTENT_TYPE = "application/avp+binary"
AVP_VERSION_HEADER = "0.2"
AVP_VERSION_STRING = "0.2.0"

# Flag bit constants
FLAG_COMPRESSED = 0x01
FLAG_HYBRID = 0x02
FLAG_HAS_MAP = 0x04
FLAG_KV_CACHE = 0x08


# --- Enums ---


class CompressionLevel(enum.Enum):
    """Zstd compression levels from the AVP spec."""

    NONE = 0
    FAST = 1
    BALANCED = 3
    MAX = 19


class PayloadType(enum.IntEnum):
    """Type of tensor payload in an AVP message."""

    HIDDEN_STATE = 0
    KV_CACHE = 1
    EMBEDDING = 2


class CommunicationMode(enum.IntEnum):
    """Communication mode negotiated during handshake."""

    LATENT = 0
    HYBRID = 1
    JSON = 2


class ProjectionMethod(enum.Enum):
    """Rosetta Stone cross-model projection methods."""

    RIDGE = "ridge"
    PROCRUSTES = "procrustes"
    VOCAB_MEDIATED = "vocab_mediated"


class DataType(enum.IntEnum):
    """Supported tensor data types."""

    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2
    INT8 = 3


# Mapping between DataType enum and numpy/string representations
_DTYPE_TO_NP = {
    DataType.FLOAT32: np.float32,
    DataType.FLOAT16: np.float16,
    DataType.INT8: np.int8,
    # BFLOAT16 has no numpy equivalent; handled via torch
}

_DTYPE_TO_STR = {
    DataType.FLOAT32: "float32",
    DataType.FLOAT16: "float16",
    DataType.BFLOAT16: "bfloat16",
    DataType.INT8: "int8",
}

_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


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
        return bool(self.flags & FLAG_COMPRESSED)

    @compressed.setter
    def compressed(self, value: bool) -> None:
        if value:
            self.flags |= FLAG_COMPRESSED
        else:
            self.flags &= ~FLAG_COMPRESSED

    @property
    def is_hybrid(self) -> bool:
        return bool(self.flags & FLAG_HYBRID)

    @property
    def has_map(self) -> bool:
        return bool(self.flags & FLAG_HAS_MAP)

    @property
    def is_kv_cache(self) -> bool:
        return bool(self.flags & FLAG_KV_CACHE)


@dataclass
class ModelIdentity:
    """Model identity exchanged during handshake."""

    model_family: str = ""
    model_id: str = ""
    model_hash: str = ""
    hidden_dim: int = 0
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    tokenizer_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model_family": self.model_family,
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
        if self.tokenizer_hash:
            d["tokenizer_hash"] = self.tokenizer_hash
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelIdentity":
        return cls(
            model_family=d.get("model_family", ""),
            model_id=d.get("model_id", ""),
            model_hash=d.get("model_hash", ""),
            hidden_dim=d.get("hidden_dim", 0),
            num_layers=d.get("num_layers", 0),
            num_kv_heads=d.get("num_kv_heads", 0),
            head_dim=d.get("head_dim", 0),
            tokenizer_hash=d.get("tokenizer_hash", ""),
        )


@dataclass
class SessionInfo:
    """Result of handshake negotiation."""

    session_id: str = ""
    mode: CommunicationMode = CommunicationMode.JSON
    local_identity: Optional[ModelIdentity] = None
    remote_identity: Optional[ModelIdentity] = None
    avp_map_id: str = ""  # non-empty if cross-model via Rosetta Stone


@dataclass
class AVPMetadata:
    """Decoded metadata from an AVP v0.2.0 message."""

    session_id: str = ""
    source_agent_id: str = ""
    target_agent_id: str = ""
    model_id: str = ""
    hidden_dim: int = 0
    num_layers: int = 0
    payload_type: PayloadType = PayloadType.HIDDEN_STATE
    dtype: DataType = DataType.FLOAT32
    tensor_shape: Tuple[int, ...] = ()
    mode: CommunicationMode = CommunicationMode.LATENT
    compression: Optional[str] = None
    confidence_score: float = 0.0
    avp_map_id: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    # Backward compatibility with v0.1.0 field names
    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim

    @property
    def data_type(self) -> str:
        return _DTYPE_TO_STR.get(self.dtype, "float32")

    @property
    def agent_id(self) -> Optional[str]:
        return self.source_agent_id or None

    @property
    def task_id(self) -> Optional[str]:
        return self.extra.get("task_id")


@dataclass
class AVPMessage:
    """A fully decoded AVP message: header + metadata + payload."""

    header: AVPHeader
    metadata: AVPMetadata
    payload: bytes = b""
    raw_size: int = 0

    @property
    def embedding(self) -> np.ndarray:
        """Backward-compatible access to payload as numpy array.

        Only valid for EMBEDDING and HIDDEN_STATE payload types.
        """
        if not self.payload:
            return np.array([], dtype=np.float32)
        dtype_str = self.metadata.data_type
        dt = np.dtype(dtype_str).newbyteorder("<")
        arr = np.frombuffer(self.payload, dtype=dt)
        return arr.copy()
