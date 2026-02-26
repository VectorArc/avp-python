"""AVP — Agent Vector Protocol Python SDK."""

from .codec import decode, encode, encode_hidden_state, encode_hybrid, encode_kv_cache
from .compression import compress, decompress
from .errors import (
    AVPError,
    DecodeError,
    EngineNotAvailableError,
    FallbackRequested,
    HandshakeError,
    IncompatibleModelsError,
    InvalidMagicError,
    RealignmentError,
    SessionError,
    SessionExpiredError,
    ShapeMismatchError,
    TransportError,
    UnsupportedVersionError,
)
from .fallback import FallbackRequest, JSONMessage
from .handshake import CompatibilityResolver, HelloMessage, compute_model_hash, compute_tokenizer_hash, extract_model_identity
from .session import Session, SessionManager
from .types import (
    ProjectionMethod,
    AVP_VERSION_HEADER,
    AVP_VERSION_STRING,
    CONTENT_TYPE,
    FLAG_COMPRESSED,
    FLAG_HAS_MAP,
    FLAG_HYBRID,
    FLAG_KV_CACHE,
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    AVPHeader,
    AVPMessage,
    AVPMetadata,
    CommunicationMode,
    CompressionLevel,
    DataType,
    ModelIdentity,
    PayloadType,
    SessionInfo,
)
from .easy import PackedMessage, pack, unpack
from .version import __version__

# Rosetta Stone (cross-model projection) — lazy-loaded because it requires torch.
_ROSETTA_NAMES = {
    "AVPMap",
    "DEFAULT_ANCHORS",
    "calibrate",
    "apply_cross_model_projection",
    "vocabulary_mediated_projection",
    "save_map",
    "load_map",
    "find_map",
    "ValidationConfig",
    "ValidationResult",
    "validate_projection",
}

# Transport classes are lazy-loaded because httpx is an optional dependency.
# Access avp.AVPClient / avp.AVPAsyncClient / avp.create_app and they'll be
# imported on first use; raises ImportError with install hint if httpx is missing.
_TRANSPORT_NAMES = {"AVPClient", "AVPAsyncClient", "create_app"}

# Connector and context classes are lazy-loaded because they require torch/transformers/vllm.
_CONNECTOR_NAMES = {"AVPContext", "HuggingFaceConnector", "VLLMConnector"}

# Easy API helpers that need lazy loading
_EASY_NAMES = {"clear_cache"}


def __getattr__(name: str):
    if name in _TRANSPORT_NAMES:
        from . import transport as _transport
        return getattr(_transport, name)
    if name in _ROSETTA_NAMES:
        from . import rosetta as _rosetta
        return getattr(_rosetta, name)
    if name == "AVPContext":
        from .context import AVPContext
        return AVPContext
    if name == "HuggingFaceConnector":
        from .connectors.huggingface import HuggingFaceConnector
        return HuggingFaceConnector
    if name == "VLLMConnector":
        from .connectors.vllm import VLLMConnector
        return VLLMConnector
    if name in _EASY_NAMES:
        from . import easy as _easy
        return getattr(_easy, name)
    raise AttributeError(f"module 'avp' has no attribute {name}")


__all__ = [
    # Codec
    "encode",
    "decode",
    "encode_hidden_state",
    "encode_kv_cache",
    "encode_hybrid",
    # Compression
    "compress",
    "decompress",
    # Transport (lazy — requires httpx)
    "AVPClient",
    "AVPAsyncClient",
    "create_app",
    # Handshake
    "HelloMessage",
    "CompatibilityResolver",
    "compute_model_hash",
    "compute_tokenizer_hash",
    "extract_model_identity",
    # Session
    "Session",
    "SessionManager",
    # Fallback
    "JSONMessage",
    "FallbackRequest",
    # Types
    "AVPHeader",
    "AVPMessage",
    "AVPMetadata",
    "ModelIdentity",
    "SessionInfo",
    "CompressionLevel",
    "PayloadType",
    "CommunicationMode",
    "DataType",
    # Constants
    "MAGIC",
    "PROTOCOL_VERSION",
    "HEADER_SIZE",
    "CONTENT_TYPE",
    "AVP_VERSION_HEADER",
    "AVP_VERSION_STRING",
    "FLAG_COMPRESSED",
    "FLAG_HYBRID",
    "FLAG_HAS_MAP",
    "FLAG_KV_CACHE",
    "ProjectionMethod",
    # Rosetta Stone (lazy — requires torch)
    "AVPMap",
    "calibrate",
    "apply_cross_model_projection",
    "vocabulary_mediated_projection",
    "save_map",
    "load_map",
    "find_map",
    "ValidationConfig",
    "ValidationResult",
    "validate_projection",
    # Connectors & Context (lazy — requires torch/transformers/vllm)
    "AVPContext",
    "HuggingFaceConnector",
    "VLLMConnector",
    # Easy API (pack/unpack)
    "pack",
    "unpack",
    "PackedMessage",
    "clear_cache",
    # Errors
    "AVPError",
    "InvalidMagicError",
    "UnsupportedVersionError",
    "DecodeError",
    "TransportError",
    "HandshakeError",
    "SessionError",
    "SessionExpiredError",
    "ShapeMismatchError",
    "RealignmentError",
    "FallbackRequested",
    "IncompatibleModelsError",
    "EngineNotAvailableError",
    # Version
    "__version__",
]
