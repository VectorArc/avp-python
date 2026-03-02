"""AVPContext — wraps a KV-cache with metadata for the high-level API."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AVPContext:
    """Latent context produced by think(), consumed by generate() or another think().

    Holds a KV-cache (tensor references, no copy) plus metadata for
    compatibility checking and optional serialization.

    Same-process usage never needs serialization — just pass the context
    object between think() and generate() calls. Use to_bytes()/from_bytes()
    only for cross-process transfer.

    Universal mode: when ``is_universal=True``, universal_tokens holds
    [K+2, D_universal] tensor instead of KV-cache. Decoded on the target
    side via KV-cache priming.
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

    # --- Universal mode fields ---

    universal_tokens: Any = None
    """Tensor [K+2, D_universal] of universal representation tokens."""

    k_tokens: int = 0
    """Number of semantic universal tokens (K, excluding special tokens)."""

    d_universal: int = 0
    """Universal space dimension."""

    is_universal: bool = False
    """True if this context carries universal tokens instead of KV-cache."""

    gate_value: float = 1.0
    """Decoder confidence gate (0-1). Applied during KV-cache priming."""

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
            AVP-encoded bytes containing KV-cache or universal tokens and metadata.
        """
        from .types import (
            AVPMetadata,
            CommunicationMode,
            DataType,
        )

        extra = {
            "model_hash": self.model_hash,
            "model_family": self.model_family,
            "num_steps": str(self.num_steps),
        }

        if self.is_universal:
            from .utils import embedding_to_bytes
            import numpy as np

            # Serialize universal tokens tensor
            tokens_np = self.universal_tokens.detach().float().cpu().numpy()
            payload = embedding_to_bytes(tokens_np)

            metadata = AVPMetadata(
                session_id=session_id,
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                model_id=model_id,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dtype=DataType.FLOAT32,
                tensor_shape=tokens_np.shape,
                mode=CommunicationMode.UNIVERSAL,
                extra=extra,
            )

            from .codec import encode_urt
            return encode_urt(
                payload, metadata,
                k_tokens=self.k_tokens,
                d_universal=self.d_universal,
            )
        else:
            from .codec import encode_kv_cache
            from .kv_cache import serialize_kv_cache
            from .types import PayloadType

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
            Restored AVPContext with DynamicCache or universal tokens.
        """
        from .codec import decode as avp_decode
        from .types import PayloadType

        msg = avp_decode(data)
        extra = msg.metadata.extra

        if msg.metadata.payload_type == PayloadType.URT:
            import numpy as np

            dtype_str = msg.metadata.data_type
            dt = np.dtype(dtype_str).newbyteorder("<")
            arr = np.frombuffer(msg.payload, dtype=dt).copy()
            shape = msg.metadata.tensor_shape
            if shape:
                arr = arr.reshape(shape)

            # Lazy-import torch only when needed
            import torch
            tokens_tensor = torch.from_numpy(arr).to(device)

            return cls(
                past_key_values=None,
                model_hash=extra.get("model_hash", ""),
                num_steps=int(extra.get("num_steps", "0")),
                seq_len=0,
                model_family=extra.get("model_family", ""),
                hidden_dim=msg.metadata.hidden_dim,
                num_layers=msg.metadata.num_layers,
                universal_tokens=tokens_tensor,
                k_tokens=int(extra.get("k_tokens", "0")),
                d_universal=int(extra.get("d_universal", "0")),
                is_universal=True,
            )
        else:
            from .kv_cache import deserialize_kv_cache, legacy_to_dynamic_cache

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
