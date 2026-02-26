"""Zero-friction pack()/unpack() API for AVP.

Progressive layers — developer only pays for what they use:

  Layer 0: JSON messaging     pack(text) / unpack(data)
  Layer 1: + Model identity   pack(text, model="Qwen/...")
  Layer 2: + Latent context   pack(text, model="Qwen/...", think_steps=20)

Messages are self-describing. No handshake required.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from .types import AVP_VERSION_HEADER, MAGIC

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
) -> PackedMessage:
    """Pack a text message for transfer between agents.

    Args:
        content: The text to send.
        model: Local model identifier (HuggingFace path/name).
        context: A previous PackedMessage whose latent context should be reused.
        think_steps: Number of latent thinking steps (Layer 2).

    Returns:
        PackedMessage — use str(msg) for text, bytes(msg) for wire format.
    """
    identity = None
    avp_context = None

    if model is not None:
        identity = _get_local_identity(model)

        if think_steps > 0:
            connector = _get_or_create_connector(model)
            prior_context = context.context if context is not None else None
            avp_context = connector.think(
                content, steps=think_steps, context=prior_context
            )

    return PackedMessage(
        content=content,
        identity=identity,
        context=avp_context,
        model=model,
    )


# ---------------------------------------------------------------------------
# unpack()
# ---------------------------------------------------------------------------


def unpack(
    data: Union[bytes, str, "PackedMessage"],
    *,
    model: Optional[str] = None,
    context: Optional[PackedMessage] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Unpack an AVP message back to text. Optionally generate a response.

    Args:
        data: Wire bytes, JSON string, raw text, or PackedMessage.
        model: Local model to use for generation. If None, just extracts text.
        context: Previous PackedMessage providing latent context for generation.
        max_new_tokens: Max tokens for generation (only used with model=).
        temperature: Sampling temperature for generation (only used with model=).

    Returns:
        The text content, or generated response if model= is provided.
    """
    packed = _decode_input(data)

    if model is None:
        return packed.content

    connector = _get_or_create_connector(model)
    avp_context = None
    if context is not None:
        avp_context = context.context
    elif packed.context is not None:
        avp_context = packed.context
    return connector.generate(
        packed.content,
        context=avp_context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Input decoding
# ---------------------------------------------------------------------------


def _decode_input(data: Union[bytes, str, "PackedMessage"]) -> PackedMessage:
    """Decode various input formats into a PackedMessage."""
    if isinstance(data, PackedMessage):
        return data

    if isinstance(data, bytes):
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
