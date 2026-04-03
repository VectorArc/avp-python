"""AVPContext — wraps a KV-cache with metadata for the high-level API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import PayloadType


@dataclass(kw_only=True)
class AVPContext:
    """Latent context produced by think(), consumed by generate() or another think().

    Holds a KV-cache (tensor references, no copy) plus metadata for
    compatibility checking and optional serialization.

    Same-process usage never needs serialization — just pass the context
    object between think() and generate() calls. Use to_bytes()/from_bytes()
    only for cross-process transfer.
    """

    past_key_values: Any
    """DynamicCache or legacy tuple of (K, V) tensors per layer."""

    model_hash: str
    """SHA-256 of the source model config, for compatibility checks."""

    num_steps: int
    """Number of latent thinking steps that produced this context."""

    seq_len: int
    """Current KV-cache sequence length (total tokens cached)."""

    model_family: str = ""
    """Model architecture family, e.g. 'qwen2', 'llama'."""

    hidden_dim: int = 0
    """Model hidden dimension."""

    num_layers: int = 0
    """Number of transformer layers."""

    last_hidden_state: Any = None
    """Last hidden state [1, D] from think() for cross-model projection."""

    engine_state: Any = None
    """Engine-specific context handle (e.g., llama_context pointer).

    Set by connectors that maintain live inference state.  Not serialized
    by ``to_bytes()`` — this is a same-process, same-lifetime reference.
    """

    engine_position: int = 0
    """Current position in the engine context's KV-cache.

    Updated by connectors after think()/generate() operations.
    """

    @property
    def payload_type(self) -> "PayloadType":
        """The effective payload type of this context."""
        from .types import PayloadType
        if self.past_key_values is not None:
            return PayloadType.KV_CACHE
        if self.last_hidden_state is not None:
            return PayloadType.HIDDEN_STATE
        raise ValueError("AVPContext has neither KV-cache nor hidden state")

    def to_bytes(
        self,
        session_id: str = "",
        source_agent_id: str = "",
        target_agent_id: str = "",
        model_id: str = "",
    ) -> bytes:
        """Serialize this context to AVP wire format for cross-process transfer.

        Args:
            session_id: AVP session identifier.
            source_agent_id: Sending agent identifier.
            target_agent_id: Receiving agent identifier.
            model_id: Model identifier string.

        Returns:
            AVP-encoded bytes containing KV-cache and metadata.
        """
        from .types import (
            AVPMetadata,
            CommunicationMode,
            DataType,
            PayloadType,
        )
        from .codec import encode_kv_cache
        from .kv_cache import serialize_kv_cache

        # These fields are stored in proto `extra` (not first-class proto
        # fields) because they are SDK-private context metadata, not part
        # of the wire format contract.  Other implementations MAY ignore
        # them.  from_bytes() reads them back for identity checking.
        extra = {
            "model_hash": self.model_hash,
            "model_family": self.model_family,
            "num_steps": str(self.num_steps),
        }

        if self.past_key_values is None:
            raise ValueError(
                "Cannot serialize a HIDDEN_STATE-only context via to_bytes(). "
                "Only KV_CACHE contexts (with past_key_values) support "
                "serialization. Use output=OutputType.KV_CACHE in think()."
            )

        kv_bytes, kv_header = serialize_kv_cache(self.past_key_values)

        # Map actual KV-cache dtype to wire DataType
        _dtype_map = {"float32": DataType.FLOAT32, "float16": DataType.FLOAT16, "bfloat16": DataType.BFLOAT16}
        actual_dtype = _dtype_map.get(kv_header.dtype, DataType.FLOAT32)

        metadata = AVPMetadata(
            session_id=session_id,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            model_id=model_id,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            payload_type=self.payload_type,
            dtype=actual_dtype,
            mode=CommunicationMode.LATENT,
            extra=extra,
        )

        return encode_kv_cache(kv_bytes, metadata)

    @classmethod
    def from_bytes(cls, data: bytes, device: str = "cpu") -> "AVPContext":
        """Deserialize AVP wire bytes back to an AVPContext.

        Note: ``last_hidden_state`` is **not** serialized and will be
        ``None`` on the restored context.  Cross-model rosetta via
        serialized contexts is not supported — use in-process transfer.

        Args:
            data: AVP-encoded bytes (from to_bytes()).
            device: Target device for tensors ('cpu', 'cuda', etc.).

        Returns:
            Restored AVPContext with DynamicCache.
        """
        from .codec import decode as avp_decode
        from .kv_cache import deserialize_kv_cache, legacy_to_dynamic_cache

        msg = avp_decode(data)
        extra = msg.metadata.extra

        legacy_kv, kv_header = deserialize_kv_cache(msg.payload, device=device)
        past_kv = legacy_to_dynamic_cache(legacy_kv)

        return cls(
            past_key_values=past_kv,
            model_hash=extra.get("model_hash", ""),
            num_steps=int(extra.get("num_steps", "0")),
            seq_len=kv_header.seq_len,
            model_family=extra.get("model_family", ""),
            hidden_dim=msg.metadata.hidden_dim,
            num_layers=msg.metadata.num_layers,
        )
