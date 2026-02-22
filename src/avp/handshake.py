"""AVP handshake negotiation protocol.

Agents exchange HelloMessage to determine communication mode:
- Same model (hash match or family+structure match) → LATENT
- Different model → JSON fallback
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .errors import HandshakeError
from .types import CommunicationMode, ModelIdentity, SessionInfo


@dataclass
class HelloMessage:
    """Message exchanged during AVP handshake."""

    agent_id: str = ""
    avp_version: str = "0.2.0"
    identity: Optional[ModelIdentity] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "avp_version": self.avp_version,
            "capabilities": self.capabilities,
        }
        if self.identity:
            d["identity"] = self.identity.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HelloMessage":
        identity = None
        if "identity" in d:
            identity = ModelIdentity.from_dict(d["identity"])
        return cls(
            agent_id=d.get("agent_id", ""),
            avp_version=d.get("avp_version", "0.2.0"),
            identity=identity,
            capabilities=d.get("capabilities", {}),
        )


def compute_model_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a model config.

    Args:
        config: Model configuration dict (from HuggingFace config.to_dict()).

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    # Sort keys for deterministic serialization
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_tokenizer_hash(tokenizer: Any) -> str:
    """Compute a deterministic SHA-256 hash of a tokenizer's vocabulary.

    Two tokenizers with the same vocabulary (same token→id mapping) produce
    the same hash, enabling vocabulary-mediated cross-model projection.

    Args:
        tokenizer: A tokenizer with get_vocab() method, or a dict-like
                   object with items().

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, "items"):
        vocab = dict(tokenizer)
    else:
        return ""

    # Sort by token string for deterministic ordering
    canonical = json.dumps(sorted(vocab.items()), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_model_identity(
    model_or_config: Any,
    tokenizer: Any = None,
) -> ModelIdentity:
    """Extract ModelIdentity from a HuggingFace model, config, or dict.

    Args:
        model_or_config: One of:
            - A HuggingFace PreTrainedModel
            - A HuggingFace PretrainedConfig
            - A dict with model config fields
        tokenizer: Optional tokenizer to compute tokenizer_hash for
                   vocabulary-mediated cross-model projection.

    Returns:
        ModelIdentity with fields populated from the config.
    """
    config_dict: Dict[str, Any]

    if isinstance(model_or_config, dict):
        config_dict = model_or_config
    elif hasattr(model_or_config, "config"):
        # PreTrainedModel
        config_dict = model_or_config.config.to_dict()
    elif hasattr(model_or_config, "to_dict"):
        # PretrainedConfig
        config_dict = model_or_config.to_dict()
    else:
        raise HandshakeError(
            f"Cannot extract model identity from {type(model_or_config).__name__}"
        )

    # Extract model family from model_type or architectures
    model_family = config_dict.get("model_type", "")

    # Extract model_id from _name_or_path
    model_id = config_dict.get("_name_or_path", "")

    # Compute hash
    model_hash = compute_model_hash(config_dict)

    # Compute tokenizer hash if tokenizer provided
    tok_hash = ""
    if tokenizer is not None:
        tok_hash = compute_tokenizer_hash(tokenizer)

    return ModelIdentity(
        model_family=model_family,
        model_id=model_id,
        model_hash=model_hash,
        hidden_dim=config_dict.get("hidden_size", 0),
        num_layers=config_dict.get("num_hidden_layers", 0),
        num_kv_heads=config_dict.get("num_key_value_heads", config_dict.get("num_attention_heads", 0)),
        head_dim=config_dict.get("head_dim", 0) or (
            config_dict.get("hidden_size", 0) // config_dict.get("num_attention_heads", 1)
            if config_dict.get("num_attention_heads", 0) > 0 else 0
        ),
        tokenizer_hash=tok_hash,
    )


class CompatibilityResolver:
    """Resolves communication mode between two agents based on their identities."""

    @staticmethod
    def resolve(local: ModelIdentity, remote: ModelIdentity) -> SessionInfo:
        """Determine communication mode based on model identities.

        Resolution rules (in order):
            1. model_hash match → LATENT (identical models)
            2. model_family + hidden_dim + num_layers match → LATENT (structural match)
            3. shared tokenizer_hash → LATENT (vocab-mediated, avp_map_id="vocab:...")
            4. Rosetta Stone .avp-map file exists → LATENT (pre-calibrated)
            5. Otherwise → JSON

        Args:
            local: Local agent's model identity.
            remote: Remote agent's model identity.

        Returns:
            SessionInfo with session_id and resolved mode.
        """
        session_id = uuid.uuid4().hex

        mode = CommunicationMode.JSON  # default
        avp_map_id = ""

        if local.model_hash and remote.model_hash and local.model_hash == remote.model_hash:
            mode = CommunicationMode.LATENT
        elif (
            local.model_family
            and local.model_family == remote.model_family
            and local.hidden_dim > 0
            and local.hidden_dim == remote.hidden_dim
            and local.num_layers > 0
            and local.num_layers == remote.num_layers
        ):
            mode = CommunicationMode.LATENT

        # Check for shared tokenizer (vocab-mediated projection)
        if (
            mode == CommunicationMode.JSON
            and local.tokenizer_hash
            and remote.tokenizer_hash
            and local.tokenizer_hash == remote.tokenizer_hash
        ):
            mode = CommunicationMode.LATENT
            avp_map_id = f"vocab:{local.tokenizer_hash[:16]}"

        # Check for Rosetta Stone map file before falling back to JSON
        if mode == CommunicationMode.JSON and local.model_hash and remote.model_hash:
            from .rosetta.registry import find_map, map_id
            if find_map(local.model_hash, remote.model_hash):
                mode = CommunicationMode.LATENT
                avp_map_id = map_id(local.model_hash, remote.model_hash)

        return SessionInfo(
            session_id=session_id,
            mode=mode,
            local_identity=local,
            remote_identity=remote,
            avp_map_id=avp_map_id,
        )
