"""AVP — Agent Vector Protocol Python SDK.

Start here:
    >>> import avp
    >>> from avp import OllamaConnector
    >>> conn = OllamaConnector.from_ollama("qwen2.5:7b")
    >>> context = avp.think("Analyze this", model=conn)
    >>> answer = avp.generate("Solve it", model=conn, context=context)

With a HuggingFace model name (auto-creates connector):
    >>> context = avp.think("Analyze this", model="Qwen/Qwen2.5-7B-Instruct")
    >>> answer = avp.generate("Solve it", model="Qwen/Qwen2.5-7B-Instruct",
    ...                        context=context)

For direct connector access:
    >>> connector = avp.HuggingFaceConnector.from_pretrained("Qwen/...")
    >>> context = connector.think("...", steps=20)
    >>> answer = connector.generate("...", context=context)
"""

# --- Easy API (start here) ---
from .easy import generate, think
from .context_store import ContextStore
from .results import GenerateResult, InspectResult, ThinkResult

# --- Protocol layer ---
from .codec import decode, encode
from .codec import encode_kv_cache  # noqa: F401
from .compression import compress, decompress  # noqa: F401
from .handshake import CompatibilityResolver, extract_model_identity
from .handshake import HelloMessage, compute_model_hash, compute_tokenizer_hash  # noqa: F401
from .session import Session, SessionManager  # noqa: F401
from .fallback import JSONMessage  # noqa: F401

# --- Types ---
from .types import (
    AVPMessage,
    AVPMetadata,
    CommunicationMode,
    CompressionLevel,
    ModelIdentity,
)
from .types import AVPHeader, DataType, OutputType, PayloadType, ProjectionMethod, SessionInfo  # noqa: F401

# --- Errors ---
from .errors import AVPError, ConfigurationError, DecodeError, HandshakeError, IncompatibleModelsError
from .errors import (  # noqa: F401
    EngineNotAvailableError,
    ProjectionError,
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

    FLAG_KV_CACHE,
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
)

from .version import __version__

# Rosetta Stone (cross-model projection) — lazy-loaded because it requires torch.
_ROSETTA_NAMES = {
    "AVPMap",
    "calibrate",
    "apply_cross_model_projection",
    "vocabulary_mediated_projection",
    "vocab_overlap_projection",
    "save_map",
    "load_map",
    "find_map",
    "map_id",
    "ValidationConfig",
    "ValidationResult",
    "validate_projection",
    "TransferQualityConfig",
    "TransferQualityResult",
    "assess_transfer",
}

# Transport classes are lazy-loaded because httpx is an optional dependency.
# Access avp.AVPClient / avp.create_app and they'll be imported on first
# use; raises ImportError with install hint if httpx is missing.
_TRANSPORT_NAMES = {"AVPClient", "create_app"}

# Easy API helpers that need lazy loading
_EASY_NAMES = {"clear_cache", "inspect", "ModelSpec"}

# Metrics classes — lazy-loaded to avoid unconditional import
_METRICS_NAMES = {
    "ThinkMetrics",
    "GenerateMetrics", "TransferDiagnostics", "DebugConfig",
}


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
    if name == "LlamaCppConnector":
        from .connectors.llamacpp import LlamaCppConnector
        return LlamaCppConnector
    if name == "OllamaConnector":
        from .connectors.ollama import OllamaConnector
        return OllamaConnector
    if name == "EngineConnector":
        from .connectors.base import EngineConnector
        return EngineConnector
    if name in _EASY_NAMES:
        from . import easy as _easy
        return getattr(_easy, name)
    if name in _METRICS_NAMES:
        from . import metrics as _metrics
        return getattr(_metrics, name)
    raise AttributeError(f"module 'avp' has no attribute {name}")


__all__ = [
    # Easy API (start here)
    "think",
    "generate",
    "inspect",
    "clear_cache",
    "ModelSpec",
    "ContextStore",
    "ThinkResult",
    "GenerateResult",
    "InspectResult",
    # Observability (lazy — stdlib only)
    "ThinkMetrics",
    "GenerateMetrics",
    "TransferDiagnostics",
    "DebugConfig",
    # Connectors (lazy — requires torch/transformers/vllm/llama-cpp-python)
    "EngineConnector",
    "HuggingFaceConnector",
    "VLLMConnector",
    "LlamaCppConnector",
    "OllamaConnector",
    "AVPContext",
    # Protocol
    "encode",
    "decode",
    "CompatibilityResolver",
    "extract_model_identity",
    "SessionManager",
    # Types
    "OutputType",
    "PayloadType",
    "ModelIdentity",
    "CommunicationMode",
    "CompressionLevel",
    "AVPMessage",
    "AVPMetadata",
    # Transport (lazy — requires httpx)
    "AVPClient",
    "create_app",
    # Cross-model / Rosetta Stone (lazy — requires torch)
    "AVPMap",
    "calibrate",
    "vocabulary_mediated_projection",
    "vocab_overlap_projection",
    "apply_cross_model_projection",
    "save_map",
    "load_map",
    "find_map",
    "map_id",
    "ValidationConfig",
    "ValidationResult",
    "validate_projection",
    "TransferQualityConfig",
    "TransferQualityResult",
    "assess_transfer",
    # Errors
    "AVPError",
    "ConfigurationError",
    "IncompatibleModelsError",
    "DecodeError",
    "HandshakeError",
    # Version
    "__version__",
]
