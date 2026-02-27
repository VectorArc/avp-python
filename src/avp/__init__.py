"""AVP — Agent Vector Protocol Python SDK.

Start here:
    >>> import avp
    >>> msg = avp.pack("Hello from agent A")
    >>> avp.unpack(msg.to_bytes())
    'Hello from agent A'

Add model identity (downloads config only, not weights):
    >>> msg = avp.pack("Hello", model="Qwen/Qwen2.5-7B-Instruct")

Add latent reasoning (requires GPU + torch):
    >>> msg = avp.pack("Analyze this", model="Qwen/...", think_steps=20)
    >>> answer = avp.unpack(msg, model="Qwen/...")

For direct connector access (advanced):
    >>> connector = avp.HuggingFaceConnector.from_pretrained("Qwen/...")
    >>> context = connector.think("...", steps=20)
    >>> answer = connector.generate("...", context=context)
"""

# --- Easy API (start here) ---
from .easy import PackedMessage, pack, unpack
from .context_store import ContextStore

# --- Protocol layer ---
from .codec import decode, encode
from .codec import encode_hidden_state, encode_hybrid, encode_kv_cache  # noqa: F401
from .compression import compress, decompress  # noqa: F401
from .handshake import CompatibilityResolver, extract_model_identity
from .handshake import HelloMessage, compute_model_hash, compute_tokenizer_hash  # noqa: F401
from .session import Session, SessionManager  # noqa: F401
from .fallback import FallbackRequest, JSONMessage  # noqa: F401

# --- Types ---
from .types import (
    AVPMessage,
    AVPMetadata,
    CommunicationMode,
    CompressionLevel,
    ModelIdentity,
)
from .types import AVPHeader, DataType, PayloadType, ProjectionMethod, SessionInfo  # noqa: F401

# --- Errors ---
from .errors import AVPError, DecodeError, HandshakeError, IncompatibleModelsError
from .errors import (  # noqa: F401
    EngineNotAvailableError,
    FallbackRequested,
    InvalidMagicError,
    RealignmentError,
    SessionError,
    SessionExpiredError,
    ShapeMismatchError,
    TransportError,
    UnsupportedVersionError,
)

# --- Wire-format constants (importable, not promoted in __all__) ---
from .types import (  # noqa: F401
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
)

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

# Metrics classes — lazy-loaded to avoid unconditional import
_METRICS_NAMES = {"PackMetrics", "UnpackMetrics", "HandshakeMetrics"}


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
    if name in _METRICS_NAMES:
        from . import metrics as _metrics
        return getattr(_metrics, name)
    raise AttributeError(f"module 'avp' has no attribute {name}")


__all__ = [
    # Easy API (start here)
    "pack",
    "unpack",
    "PackedMessage",
    "clear_cache",
    "ContextStore",
    # Observability (lazy — stdlib only)
    "PackMetrics",
    "UnpackMetrics",
    "HandshakeMetrics",
    # Connectors (lazy — requires torch/transformers/vllm)
    "HuggingFaceConnector",
    "VLLMConnector",
    "AVPContext",
    # Protocol
    "encode",
    "decode",
    "CompatibilityResolver",
    "extract_model_identity",
    "SessionManager",
    # Types
    "ModelIdentity",
    "CommunicationMode",
    "CompressionLevel",
    "AVPMessage",
    "AVPMetadata",
    # Transport (lazy — requires httpx)
    "AVPClient",
    "AVPAsyncClient",
    "create_app",
    # Cross-model / Rosetta Stone (lazy — requires torch)
    "AVPMap",
    "calibrate",
    "vocabulary_mediated_projection",
    "validate_projection",
    # Errors
    "AVPError",
    "IncompatibleModelsError",
    "DecodeError",
    "HandshakeError",
    # Version
    "__version__",
]
