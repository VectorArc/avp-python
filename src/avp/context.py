"""AVPContext — wraps a KV-cache with metadata for the high-level API."""

from dataclasses import dataclass
from typing import Any


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

        extra = {
            "model_hash": self.model_hash,
            "model_family": self.model_family,
            "num_steps": str(self.num_steps),
        }

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
            payload_type=PayloadType.KV_CACHE,
            dtype=actual_dtype,
            mode=CommunicationMode.LATENT,
            extra=extra,
        )

        return encode_kv_cache(kv_bytes, metadata)

    @classmethod
    def from_bytes(cls, data: bytes, device: str = "cpu") -> "AVPContext":
        """Deserialize AVP wire bytes back to an AVPContext.

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
