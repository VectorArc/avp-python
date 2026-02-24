"""AVPContext — wraps a KV-cache with metadata for the high-level API."""

from dataclasses import dataclass
from typing import Any


@dataclass
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
            AVP-encoded bytes containing the KV-cache and metadata.
        """
        from .codec import encode_kv_cache
        from .kv_cache import serialize_kv_cache
        from .types import (
            AVPMetadata,
            CommunicationMode,
            DataType,
            PayloadType,
        )

        kv_bytes, _ = serialize_kv_cache(self.past_key_values)

        metadata = AVPMetadata(
            session_id=session_id,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            model_id=model_id,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            payload_type=PayloadType.KV_CACHE,
            dtype=DataType.FLOAT32,
            mode=CommunicationMode.LATENT,
            extra={
                "model_hash": self.model_hash,
                "model_family": self.model_family,
                "num_steps": str(self.num_steps),
            },
        )

        return encode_kv_cache(kv_bytes, metadata)

    @classmethod
    def from_bytes(cls, data: bytes, device: str = "cpu") -> "AVPContext":
        """Deserialize AVP wire bytes back to an AVPContext.

        Args:
            data: AVP-encoded bytes (from to_bytes()).
            device: Target device for tensors ('cpu', 'cuda', etc.).

        Returns:
            Restored AVPContext with DynamicCache on the specified device.
        """
        from .codec import decode as avp_decode
        from .kv_cache import deserialize_kv_cache, legacy_to_dynamic_cache

        msg = avp_decode(data)
        legacy_kv, kv_header = deserialize_kv_cache(msg.payload, device=device)
        past_kv = legacy_to_dynamic_cache(legacy_kv)

        extra = msg.metadata.extra
        return cls(
            past_key_values=past_kv,
            model_hash=extra.get("model_hash", ""),
            num_steps=int(extra.get("num_steps", "0")),
            seq_len=kv_header.seq_len,
            model_family=extra.get("model_family", ""),
            hidden_dim=msg.metadata.hidden_dim,
            num_layers=msg.metadata.num_layers,
        )
