"""Zero-friction pack()/unpack() API for AVP.

This is the recommended entry point. Progressive layers — you only pay
for what you use:

  Layer 0: JSON messaging     pack(text) / unpack(data)        (no optional deps)
  Layer 1: + Model identity   pack(text, model="Qwen/...")     (+ transformers)
  Layer 2: + Latent context   pack(..., think_steps=20)        (+ torch)

Messages are self-describing (first-byte detection). No handshake required.

When to use pack()/unpack() vs HuggingFaceConnector directly:
  - pack()/unpack(): Simple integration, gradual opt-in to latent features.
    Manages connector lifecycle, caching, and identity extraction for you.
  - HuggingFaceConnector: Direct control over model loading, device placement,
    batch generation, and AVPContext serialization. Use when you need to manage
    the model yourself or integrate with custom orchestration.
"""

import json
import logging
import time as _time
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from .types import AVP_VERSION_HEADER, MAGIC

if TYPE_CHECKING:
    from .metrics import GenerateMetrics, PackMetrics, UnpackMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PackedMessage
# ---------------------------------------------------------------------------


@dataclass
class PackedMessage:
    """A packed AVP message. Strings in, strings out.

    str(msg) returns the text content. bytes(msg) returns wire bytes.
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

        Use this when you need access to identity or context, not just text.

        Example::

            msg = PackedMessage.from_wire(wire_bytes)
            print(msg.content)    # text
            print(msg.identity)   # model identity dict (if present)
            print(msg.context)    # AVPContext (if binary with latent data)
        """
        return _decode_input(data)


# ---------------------------------------------------------------------------
# Identity helpers (Layer 1)
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
# Connector caching (Layer 2)
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
# pack()
# ---------------------------------------------------------------------------


def pack(
    content: str,
    *,
    model: Optional[str] = None,
    context: Optional[PackedMessage] = None,
    think_steps: int = 0,
    collect_metrics: bool = False,
) -> Union[PackedMessage, Tuple[PackedMessage, "PackMetrics"]]:
    """Pack a text message for transfer between agents.

    Examples::

        msg = avp.pack("Hello")                              # Layer 0: JSON
        msg = avp.pack("Hello", model="Qwen/Qwen2.5-7B")    # Layer 1: + identity
        msg = avp.pack("Hello", model="Qwen/...",            # Layer 2: + latent
                        think_steps=20)

        # With metrics:
        msg, metrics = avp.pack("Hello", collect_metrics=True)

    Args:
        content: The text to send.
        model: HuggingFace model name/path (self-hosted, local weights).
            Layer 1 downloads only the config (~1 KB) for identity.
            Layer 2 (think_steps > 0) loads the full model (cached after first call).
        context: A previous PackedMessage whose latent context should be reused.
        think_steps: Number of latent thinking steps. 0 = no latent (Layer 0/1).
            20 is the recommended value (accuracy plateaus beyond this).
        collect_metrics: If True, return ``(PackedMessage, PackMetrics)`` tuple.

    Returns:
        PackedMessage — use str(msg) for text, bytes(msg) for wire format.
        If collect_metrics=True, returns (PackedMessage, PackMetrics).
    """
    t_start = _time.perf_counter()

    if not isinstance(content, str):
        raise TypeError(f"pack() content must be str, got {type(content).__name__}")
    if think_steps > 0 and model is None:
        raise ValueError("think_steps requires model= (e.g. model='Qwen/Qwen2.5-7B-Instruct')")

    identity = None
    avp_context = None
    identity_duration = 0.0
    think_duration = 0.0

    if model is not None:
        t_id = _time.perf_counter()
        identity = _get_local_identity(model)
        identity_duration = _time.perf_counter() - t_id

        if think_steps > 0:
            connector = _get_or_create_connector(model)
            prior_context = context.context if context is not None else None
            t_think = _time.perf_counter()
            avp_context = connector.think(
                content, steps=think_steps, context=prior_context
            )
            think_duration = _time.perf_counter() - t_think
            logger.info(
                "pack() latent thinking: steps=%d duration=%.3fs",
                think_steps, think_duration,
            )

    layer = 0 if model is None else (2 if think_steps > 0 else 1)
    logger.debug(
        "pack() layer=%d model=%s think_steps=%d",
        layer, model, think_steps,
    )

    result = PackedMessage(
        content=content,
        identity=identity,
        context=avp_context,
        model=model,
    )

    if collect_metrics:
        from .metrics import PackMetrics

        metrics = PackMetrics(
            layer=layer,
            model=model,
            think_steps=think_steps,
            has_prior_context=context is not None,
            duration_s=_time.perf_counter() - t_start,
            identity_duration_s=identity_duration,
            think_duration_s=think_duration,
        )
        return result, metrics

    return result


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def generate(
    content: str,
    *,
    model: str,
    think_steps: int = 20,
    store: Optional[Any] = None,
    store_key: Optional[str] = None,
    prior_key: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    collect_metrics: bool = False,
) -> Union[str, Tuple[str, "GenerateMetrics"]]:
    """Think about a prompt, optionally store/retrieve context, and generate text.

    Combines pack() + ContextStore + unpack() into a single call.  This is
    the pattern every agent framework integration needs::

        # Before (7 lines):
        packed = avp.pack(prompt, model=M, think_steps=20, context=prior)
        store.store("researcher", packed)
        text = avp.unpack(packed, model=M, max_new_tokens=256)

        # After (1 line):
        text = avp.generate(prompt, model=M, store=store,
                            store_key="researcher")

    Without a store, it still collapses the pack→unpack dance::

        text = avp.generate(prompt, model=M)

    Args:
        content: The prompt text.
        model: HuggingFace model name/path (required — generate always
            needs a model).
        think_steps: Number of latent thinking steps.  Defaults to 20
            (the recommended value).  Set to 0 for text-only generation
            without latent context.
        store: A ``ContextStore`` instance for automatic context management.
            Required if *store_key* or *prior_key* is set.
        store_key: If set, the packed context is stored under this key
            after thinking (requires *store*).
        prior_key: If set, retrieve prior context from *store* under this
            key and pass it as input context (requires *store*).
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

    # Retrieve prior context from store
    prior_context: Optional[PackedMessage] = None
    if prior_key is not None and store is not None:
        prior_context = store.get(prior_key)

    # Pack (think)
    t_think = _time.perf_counter()
    packed = pack(
        content,
        model=model,
        context=prior_context,
        think_steps=think_steps,
    )
    think_duration = _time.perf_counter() - t_think

    # Store
    stored = False
    if store_key is not None and store is not None:
        store.store(store_key, packed)
        stored = True

    # Generate text
    connector = _get_or_create_connector(model)
    avp_context = packed.context

    t_gen = _time.perf_counter()
    text = connector.generate(
        content,
        context=avp_context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    generate_duration = _time.perf_counter() - t_gen

    logger.debug(
        "generate() model=%s think_steps=%d stored=%s prior=%s",
        model, think_steps, store_key, prior_key,
    )
    logger.info(
        "generate() think=%.3fs generate=%.3fs total=%.3fs",
        think_duration, generate_duration, _time.perf_counter() - t_start,
    )

    if collect_metrics:
        from .metrics import GenerateMetrics

        metrics = GenerateMetrics(
            model=model,
            think_steps=think_steps,
            has_prior_context=prior_context is not None,
            stored=stored,
            duration_s=_time.perf_counter() - t_start,
            think_duration_s=think_duration,
            generate_duration_s=generate_duration,
        )
        return text, metrics

    return text


# ---------------------------------------------------------------------------
# unpack()
# ---------------------------------------------------------------------------


def unpack(
    data: Union[bytes, bytearray, memoryview, str, "PackedMessage"],
    *,
    model: Optional[str] = None,
    context: Optional[PackedMessage] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    collect_metrics: bool = False,
) -> Union[str, Tuple[str, "UnpackMetrics"]]:
    """Unpack an AVP message back to text. Optionally generate a response.

    Accepts any format — AVP binary, AVP JSON, legacy JSONMessage, plain text,
    or a PackedMessage object. Format is detected automatically.

    Examples::

        text = avp.unpack(wire_bytes)                          # extract text
        text = avp.unpack("plain string")                      # passthrough
        answer = avp.unpack(msg, model="Qwen/Qwen2.5-7B")     # generate response

        # With metrics:
        text, metrics = avp.unpack("hello", collect_metrics=True)

    Args:
        data: Wire bytes, JSON string, raw text, or PackedMessage.
        model: HuggingFace model name/path for generation. If None, just
            extracts text (no model needed, no GPU needed).
        context: Previous PackedMessage providing latent context for generation.
        max_new_tokens: Max tokens for generation (only used with model=).
        temperature: Sampling temperature for generation (only used with model=).
        collect_metrics: If True, return ``(str, UnpackMetrics)`` tuple.

    Returns:
        The text content, or generated response if model= is provided.
        If collect_metrics=True, returns (str, UnpackMetrics).
    """
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
        has_ctx = avp_context is not None
        logger.info(
            "unpack() generated: has_context=%s duration=%.3fs",
            has_ctx, generate_duration,
        )

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


def _detect_format(data: Union[bytes, bytearray, memoryview, str, "PackedMessage"]) -> str:
    """Determine input format label for metrics."""
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


# ---------------------------------------------------------------------------
# Input decoding
# ---------------------------------------------------------------------------


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

    raise TypeError(f"unpack() expects bytes, str, or PackedMessage, got {type(data).__name__}")


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
