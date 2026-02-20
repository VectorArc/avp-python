"""AVP — Agent Vector Protocol Python SDK."""

from .codec import decode, encode, encode_hidden_state, encode_kv_cache
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
from .handshake import CompatibilityResolver, HelloMessage, compute_model_hash, extract_model_identity
from .session import Session, SessionManager
from .types import (
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
from .version import __version__

# Transport classes are lazy-loaded because httpx is an optional dependency.
# Access avp.AVPClient / avp.AVPAsyncClient / avp.create_app and they'll be
# imported on first use; raises ImportError with install hint if httpx is missing.
_TRANSPORT_NAMES = {"AVPClient", "AVPAsyncClient", "create_app"}


def __getattr__(name: str):
    if name in _TRANSPORT_NAMES:
        from . import transport as _transport
        return getattr(_transport, name)
    raise AttributeError(f"module 'avp' has no attribute {name}")


__all__ = [
    # Codec
    "encode",
    "decode",
    "encode_hidden_state",
    "encode_kv_cache",
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
