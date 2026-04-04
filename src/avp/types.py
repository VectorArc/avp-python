"""AVP data types and dataclasses."""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# --- Constants ---
#
# Version semantics:
#   PROTOCOL_VERSION (header byte 2) — identifies the fixed header format.
#     Changes only if the 12-byte header layout changes (nuclear option).
#     Decoders MUST reject messages with unknown version bytes.
#   AVP_VERSION_HEADER — transport binding version (HTTP AVP-Version header).
#   AVP_VERSION_STRING — SDK/protocol feature version (handshake avp_version).
#     Used for capability advertisement, not wire format gating.

MAGIC = b"\x41\x56"  # "AV"
PROTOCOL_VERSION = 0x01
HEADER_SIZE = 12  # 2 magic + 1 version + 1 flags + 4 payload_len + 4 metadata_len

CONTENT_TYPE = "application/avp+binary"
AVP_VERSION_HEADER = "0.4"
AVP_VERSION_STRING = "0.6.0"

# Flag bit constants — fast-path routing hints.
# If FLAG_KV_CACHE is set, metadata.payload_type MUST be KV_CACHE.
# Decoders SHOULD validate consistency; metadata is authoritative on conflict.
FLAG_COMPRESSED = 0x01
FLAG_HAS_MAP = 0x02
FLAG_KV_CACHE = 0x04


# --- Enums ---


class CompressionLevel(enum.Enum):
    """Zstd compression levels from the AVP spec."""

    NONE = 0
    FAST = 1
    BALANCED = 3
    MAX = 19


class OutputType(enum.Enum):
    """Requested output type for :meth:`think`.

    Controls what the connector includes in the returned context.
    This is an API-level hint — it never appears on the wire.

    Use as the ``output=`` parameter on ``think()``::

        result = avp.think("prompt", model=conn, output=OutputType.AUTO)
    """

    AUTO = "auto"
    """Let the system decide (default).  Currently resolves to
    ``KV_CACHE`` for same-model and ``HIDDEN_STATE`` for cross-model."""

    KV_CACHE = "kv_cache"
    """Full KV-cache + hidden state.  Best for same-model,
    same-process transfer."""

    HIDDEN_STATE = "hidden_state"
    """Only the last hidden state ``[1, D]``.  KV-cache is freed
    immediately, reducing VRAM."""

    def resolve(self) -> "PayloadType":
        """Resolve this output request to a concrete wire PayloadType.

        ``AUTO`` and ``KV_CACHE`` resolve to ``PayloadType.KV_CACHE``.
        ``HIDDEN_STATE`` resolves to ``PayloadType.HIDDEN_STATE``.
        """
        if self == OutputType.HIDDEN_STATE:
            return PayloadType.HIDDEN_STATE
        return PayloadType.KV_CACHE


class PayloadType(enum.IntEnum):
    """Payload type in AVP wire format.

    Values correspond to the proto schema (field 7 of Metadata).
    This enum only contains wire-valid values — it never carries
    API-level hints like "auto".
    """

    HIDDEN_STATE = 0
    KV_CACHE = 1


class CommunicationMode(enum.IntEnum):
    """Communication mode negotiated during handshake."""

    LATENT = 0
    JSON = 1  # Proto uses JSON_MODE (reserved word workaround); wire value is the same

    def __str__(self) -> str:
        return self.name


class ProjectionMethod(enum.Enum):
    """Rosetta Stone cross-model projection methods."""

    VOCAB_MEDIATED = "vocab_mediated"
    VOCAB_OVERLAP = "vocab_overlap"


class DataType(enum.IntEnum):
    """Supported tensor data types.

    Values 4-5 are reserved for future quantized types. Decoders
    encountering an unknown value MUST reject the message.
    """

    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2
    INT8 = 3
    # Reserved: FLOAT8 = 4, INT4 = 5


# Mapping between DataType enum and string representations
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
    resolution_path: str = ""  # e.g. "hash_match", "structural_match", "json_fallback"


@dataclass
class AVPMetadata:
    """Decoded metadata from an AVP message."""

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
    avp_map_id: str = ""
    extra: Dict[str, str] = field(default_factory=dict)



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

        Only valid for HIDDEN_STATE payload types.
        """
        if not self.payload:
            return np.array([], dtype=np.float32)
        dtype_str = _DTYPE_TO_STR.get(self.metadata.dtype, "float32")
        dt = np.dtype(dtype_str).newbyteorder("<")
        arr = np.frombuffer(self.payload, dtype=dt)
        return arr.copy()
