"""Easy API: avp.think() and avp.generate().

    import avp

    # Get latent context
    context = avp.think("Analyze this problem", model="Qwen/Qwen2.5-7B-Instruct")

    # Generate with latent context
    answer = avp.generate("Solve it", model="Qwen/Qwen2.5-7B-Instruct", context=context)

    # Or do both in one call
    answer = avp.generate("Analyze and solve: 24*17+3", model="Qwen/Qwen2.5-7B-Instruct")

For direct connector access (advanced):
    connector = avp.HuggingFaceConnector.from_pretrained("Qwen/...")
    context = connector.think("...", steps=20)
    answer = connector.generate("...", context=context)
"""

import json
import logging
import time as _time
import threading
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from .types import AVP_VERSION_HEADER, MAGIC

if TYPE_CHECKING:
    from .context import AVPContext
    from .metrics import GenerateMetrics, ThinkMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PackedMessage (DEPRECATED — kept for backward compatibility)
# ---------------------------------------------------------------------------


@dataclass
class PackedMessage:
    """A packed AVP message.

    .. deprecated:: 0.3.0
        Use ``avp.think()`` to create latent context (returns ``AVPContext``).
    """

    content: str
    """The text content of the message."""

    identity: Optional[Dict[str, Any]] = None
    """Model identity dict (Layer 1+). None for plain text."""

    context: Any = field(default=None, repr=False)
    """AVPContext for latent transfer (Layer 2). None for JSON-only."""

    model: Optional[str] = None
    """Original model string passed to pack()."""

    def to_bytes(self) -> bytes:
        """Serialize to wire format (JSON or AVP binary)."""
        if self.context is not None:
            return self.context.to_bytes(model_id=self.model or "")
        d: Dict[str, Any] = {"avp": AVP_VERSION_HEADER, "content": self.content}
        if self.identity is not None:
            d["identity"] = self.identity
        return json.dumps(d).encode("utf-8")

    def __str__(self) -> str:
        return self.content

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    @classmethod
    def from_wire(cls, data: Union[bytes, bytearray, memoryview, str]) -> "PackedMessage":
        """Decode wire data into a PackedMessage for inspection.

        """
        return _decode_input(data)


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

_identity_cache: Dict[str, Dict[str, Any]] = {}
_identity_lock = threading.Lock()


def _get_local_identity(model_name: str) -> Optional[Dict[str, Any]]:
    """Extract model identity from HuggingFace config (downloads config only, not weights).

    Returns None if transformers is not installed.
    """
    with _identity_lock:
        if model_name in _identity_cache:
            return _identity_cache[model_name]

    try:
        from transformers import AutoConfig
        from .handshake import compute_model_hash
    except ImportError:
        return {"model_id": model_name}

    try:
        config = AutoConfig.from_pretrained(model_name)
        cfg = config.to_dict()
        hidden_dim = cfg.get("hidden_size", 0)
        num_layers = cfg.get("num_hidden_layers", 0)
        model_family = cfg.get("model_type", "")
        model_hash = compute_model_hash(cfg)

        identity = {
            "model_hash": model_hash,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "model_family": model_family,
            "model_id": model_name,
        }
    except Exception:
        logger.warning("Failed to extract identity for %s", model_name, exc_info=True)
        identity = {"model_id": model_name}

    with _identity_lock:
        _identity_cache[model_name] = identity
    return identity


# ---------------------------------------------------------------------------
# Connector caching
# ---------------------------------------------------------------------------

_connector_cache: Dict[str, Any] = {}
_connector_lock = threading.Lock()


def _get_or_create_connector(model_name: str) -> Any:
    """Get or create a cached HuggingFaceConnector for latent operations."""
    with _connector_lock:
        if model_name in _connector_cache:
            return _connector_cache[model_name]

    from .connectors.huggingface import HuggingFaceConnector

    connector = HuggingFaceConnector.from_pretrained(model_name)

    with _connector_lock:
        _connector_cache[model_name] = connector
    return connector


def clear_cache() -> None:
    """Free cached connectors and identities to reclaim memory."""
    with _connector_lock:
        _connector_cache.clear()
    with _identity_lock:
        _identity_cache.clear()


# ---------------------------------------------------------------------------
# think()
# ---------------------------------------------------------------------------


def think(
    prompt: str,
    *,
    model: str,
    steps: int = 20,
    context: Optional["AVPContext"] = None,
    collect_metrics: bool = False,
) -> Union["AVPContext", Tuple["AVPContext", "ThinkMetrics"]]:
    """Run latent thinking steps on a prompt. Returns AVPContext.

    Equivalent to creating a HuggingFaceConnector and calling
    ``connector.think()``, but handles model loading and caching for you.

    Examples::

        context = avp.think("Analyze this", model="Qwen/Qwen2.5-7B-Instruct")
        context = avp.think("Continue", model="Qwen/...", context=prior_context)

    Args:
        prompt: The text prompt to think about.
        model: HuggingFace model name/path (required).
        steps: Number of latent thinking steps. Default 20.
        context: Prior AVPContext to continue from.
        collect_metrics: If True, return ``(AVPContext, ThinkMetrics)`` tuple.

    Returns:
        AVPContext containing the KV-cache and metadata.
        If collect_metrics=True, returns (AVPContext, ThinkMetrics).
    """
    t_start = _time.perf_counter()

    if not isinstance(prompt, str):
        raise TypeError(f"think() prompt must be str, got {type(prompt).__name__}")

    connector = _get_or_create_connector(model)
    t_think = _time.perf_counter()
    avp_context = connector.think(prompt, steps=steps, context=context)
    think_duration = _time.perf_counter() - t_think

    logger.info(
        "think() model=%s steps=%d duration=%.3fs",
        model, steps, think_duration,
    )

    if collect_metrics:
        from .metrics import ThinkMetrics

        metrics = ThinkMetrics(
            model=model,
            steps=steps,
            has_prior_context=context is not None,
            duration_s=_time.perf_counter() - t_start,
            think_duration_s=think_duration,
        )
        return avp_context, metrics

    return avp_context


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def generate(
    content: str,
    *,
    model: str,
    source_model: Optional[str] = None,
    steps: int = 20,
    context: Optional["AVPContext"] = None,
    store: Optional[Any] = None,
    store_key: Optional[str] = None,
    prior_key: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    collect_metrics: bool = False,
) -> Union[str, Tuple[str, "GenerateMetrics"]]:
    """Think about a prompt, optionally store/retrieve context, and generate text.

    One-liner for the common think + generate pattern::

        text = avp.generate("Solve: 2+2", model="Qwen/Qwen2.5-7B-Instruct")

    Cross-model (Rosetta projection)::

        text = avp.generate(prompt, model="target", source_model="source")

    With context store for multi-turn::

        text = avp.generate(prompt, model=M, store=store, store_key="agent-a")

    Args:
        content: The prompt text.
        model: HuggingFace model name/path (required).
        source_model: Source model for cross-model projection. When set,
            thinks on the source model and projects to the target (``model``)
            via Rosetta Stone.
        steps: Number of latent thinking steps. Default 20. Set to 0 for
            text-only generation without latent context.
        context: Prior AVPContext to continue from.
        store: A ``ContextStore`` instance for automatic context management.
        store_key: Store the context under this key after thinking.
        prior_key: Retrieve prior context from store under this key.
        max_new_tokens: Max tokens for generation.
        temperature: Sampling temperature.
        collect_metrics: If True, return ``(str, GenerateMetrics)`` tuple.

    Returns:
        Generated text response.
        If collect_metrics=True, returns (str, GenerateMetrics).
    """
    t_start = _time.perf_counter()

    if not isinstance(content, str):
        raise TypeError(f"generate() content must be str, got {type(content).__name__}")
    if (store_key is not None or prior_key is not None) and store is None:
        raise ValueError("store_key/prior_key require store= (pass a ContextStore)")

    # Cross-model path: think on source, project to target
    if source_model is not None:
        source_connector = _get_or_create_connector(source_model)
        target_connector = _get_or_create_connector(model)

        t_think = _time.perf_counter()
        source_context = source_connector.think(content, steps=steps)
        think_duration = _time.perf_counter() - t_think

        t_gen = _time.perf_counter()
        text = target_connector.generate(
            content,
            context=source_context,
            source=source_connector,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        generate_duration = _time.perf_counter() - t_gen

        logger.info(
            "generate() cross-model %s→%s think=%.3fs generate=%.3fs total=%.3fs",
            source_model, model, think_duration, generate_duration,
            _time.perf_counter() - t_start,
        )

        if collect_metrics:
            from .metrics import GenerateMetrics

            metrics = GenerateMetrics(
                model=model,
                steps=steps,
                has_prior_context=False,
                stored=False,
                duration_s=_time.perf_counter() - t_start,
                think_duration_s=think_duration,
                generate_duration_s=generate_duration,
            )
            return text, metrics

        return text

    # Same-model path

    # Retrieve prior context from store
    if prior_key is not None and store is not None:
        prior = store.get(prior_key)
        if prior is not None:
            context = prior

    # Think
    t_think = _time.perf_counter()
    if steps > 0:
        avp_context = think(content, model=model, steps=steps, context=context)
    else:
        avp_context = context
    think_duration = _time.perf_counter() - t_think

    # Store context
    stored = False
    if store_key is not None and store is not None and avp_context is not None:
        store.store(store_key, avp_context)
        stored = True

    # Generate text
    connector = _get_or_create_connector(model)

    t_gen = _time.perf_counter()
    text = connector.generate(
        content,
        context=avp_context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    generate_duration = _time.perf_counter() - t_gen

    logger.debug(
        "generate() model=%s steps=%d stored=%s prior=%s",
        model, steps, store_key, prior_key,
    )
    logger.info(
        "generate() think=%.3fs generate=%.3fs total=%.3fs",
        think_duration, generate_duration, _time.perf_counter() - t_start,
    )

    if collect_metrics:
        from .metrics import GenerateMetrics

        metrics = GenerateMetrics(
            model=model,
            steps=steps,
            has_prior_context=context is not None,
            stored=stored,
            duration_s=_time.perf_counter() - t_start,
            think_duration_s=think_duration,
            generate_duration_s=generate_duration,
        )
        return text, metrics

    return text


# ---------------------------------------------------------------------------
# Deprecated: pack() / unpack()
# ---------------------------------------------------------------------------


def pack(
    content: str,
    *,
    model: Optional[str] = None,
    context: Optional[PackedMessage] = None,
    think_steps: int = 0,
    universal: bool = False,
    rollout_steps: int = 256,
    collect_metrics: bool = False,
) -> Union[PackedMessage, Tuple[PackedMessage, Any]]:
    """Pack a text message for transfer between agents.

    .. deprecated:: 0.3.0
        Use ``avp.think()`` for latent context or ``avp.generate()`` for text.
    """
    warnings.warn(
        "avp.pack() is deprecated and will be removed in v0.4. "
        "Use avp.think() for latent context or avp.generate() for text output.",
        DeprecationWarning,
        stacklevel=2,
    )
    t_start = _time.perf_counter()

    if not isinstance(content, str):
        raise TypeError(f"pack() content must be str, got {type(content).__name__}")
    if think_steps > 0 and model is None:
        raise ValueError("think_steps requires model= (e.g. model='Qwen/Qwen2.5-7B-Instruct')")
    if universal and model is None:
        raise ValueError("universal=True requires model= (e.g. model='Qwen/Qwen2.5-7B-Instruct')")

    identity = None
    avp_context = None
    think_duration = 0.0

    if model is not None:
        identity = _get_local_identity(model)

        if universal:
            warnings.warn(
                "universal=True is deprecated and will be removed in a future version. "
                "Universal KV-cache priming via inputs_embeds has been validated negative "
                "on text-only LLMs (0% same-model accuracy). "
                "Use think_steps for same-model or rosetta projection for cross-model.",
                DeprecationWarning,
                stacklevel=2,
            )
            connector = _get_or_create_connector(model)
            t_think = _time.perf_counter()
            avp_context = connector.encode_universal(content, steps=rollout_steps)
            think_duration = _time.perf_counter() - t_think
        elif think_steps > 0:
            connector = _get_or_create_connector(model)
            prior_context = context.context if context is not None else None
            t_think = _time.perf_counter()
            avp_context = connector.think(
                content, steps=think_steps, context=prior_context
            )
            think_duration = _time.perf_counter() - t_think

    result = PackedMessage(
        content=content,
        identity=identity,
        context=avp_context,
        model=model,
    )

    if collect_metrics:
        from .metrics import ThinkMetrics

        metrics = ThinkMetrics(
            model=model,
            steps=think_steps,
            has_prior_context=context is not None,
            duration_s=_time.perf_counter() - t_start,
            think_duration_s=think_duration,
        )
        return result, metrics

    return result


def unpack(
    data: Union[bytes, bytearray, memoryview, str, "PackedMessage"],
    *,
    model: Optional[str] = None,
    context: Optional[PackedMessage] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    collect_metrics: bool = False,
) -> Union[str, Tuple[str, Any]]:
    """Unpack an AVP message back to text. Optionally generate a response.

    .. deprecated:: 0.3.0
        Use ``avp.generate()`` for text generation.
    """
    warnings.warn(
        "avp.unpack() is deprecated and will be removed in v0.4. "
        "Use avp.generate() for text generation.",
        DeprecationWarning,
        stacklevel=2,
    )
    t_start = _time.perf_counter()

    input_format = _detect_format(data)

    t_decode = _time.perf_counter()
    packed = _decode_input(data)
    decode_duration = _time.perf_counter() - t_decode

    logger.debug("unpack() format=%s model=%s", input_format, model)

    generate_duration = 0.0
    if model is None:
        text = packed.content
    else:
        connector = _get_or_create_connector(model)
        avp_context = None
        if context is not None:
            avp_context = context.context
        elif packed.context is not None:
            avp_context = packed.context

        t_gen = _time.perf_counter()
        text = connector.generate(
            packed.content,
            context=avp_context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        generate_duration = _time.perf_counter() - t_gen

    if collect_metrics:
        from .metrics import UnpackMetrics

        has_context = False
        if context is not None and context.context is not None:
            has_context = True
        elif packed.context is not None:
            has_context = True

        metrics = UnpackMetrics(
            input_format=input_format,
            has_context=has_context,
            generated=model is not None,
            duration_s=_time.perf_counter() - t_start,
            decode_duration_s=decode_duration,
            generate_duration_s=generate_duration,
        )
        return text, metrics

    return text


# ---------------------------------------------------------------------------
# Internal decode helpers
# ---------------------------------------------------------------------------


def _detect_format(data: Union[bytes, bytearray, memoryview, str, "PackedMessage"]) -> str:
    """Determine input format label."""
    if isinstance(data, PackedMessage):
        return "packed_message"
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data) if not isinstance(data, bytes) else data
        if raw[:2] == MAGIC:
            return "binary"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return "binary"
        if text.lstrip().startswith("{"):
            return "json"
        return "text"
    if isinstance(data, str):
        if data.lstrip().startswith("{"):
            return "json"
        return "text"
    return "unknown"


def _decode_input(data: Union[bytes, str, "PackedMessage"]) -> PackedMessage:
    """Decode various input formats into a PackedMessage."""
    if isinstance(data, PackedMessage):
        return data

    if isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
        # First-byte detection
        if data[:2] == MAGIC:
            return _decode_avp_binary(data)
        # Try JSON
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Cannot decode data: not valid AVP binary or UTF-8 text")
        return _decode_json_or_text(text)

    if isinstance(data, str):
        return _decode_json_or_text(data)

    raise TypeError(f"Expected bytes or str, got {type(data).__name__}")


def _decode_json_or_text(text: str) -> PackedMessage:
    """Try to parse as JSON (AVP or legacy JSONMessage), fall back to raw text."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            d = json.loads(stripped)
        except json.JSONDecodeError:
            return PackedMessage(content=text)

        # New AVP easy format: {"avp": "0.2", "content": "..."}
        if "avp" in d:
            return PackedMessage(
                content=d.get("content", ""),
                identity=d.get("identity"),
            )
        # Legacy JSONMessage format: {"avp_version": "0.2.0", "content": "..."}
        if "avp_version" in d:
            return PackedMessage(content=d.get("content", ""))
        # Unknown JSON — return as-is
        return PackedMessage(content=text)

    return PackedMessage(content=text)


def _decode_avp_binary(data: bytes) -> PackedMessage:
    """Decode AVP binary wire format."""
    from .codec import decode as avp_decode

    msg = avp_decode(data)
    text = msg.text_fallback or ""

    # If it's a hybrid message with latent content, reconstruct AVPContext
    avp_context = None
    if msg.payload:
        try:
            from .context import AVPContext

            avp_context = AVPContext.from_bytes(data)
        except Exception:
            logger.warning("Failed to reconstruct AVPContext from binary", exc_info=True)

    return PackedMessage(content=text, context=avp_context)
