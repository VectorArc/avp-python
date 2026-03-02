"""Universal adapter storage and registry.

Manages trained adapter files on disk — save, load, and find universal
encoder/decoder adapters identified by model config hash.

Pattern follows rosetta/registry.py (save_map/load_map/find_map).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .._torch_compat import require_torch as _require_torch
from .config import UniversalConfig


_ADAPTER_DIR = Path(os.environ.get("AVP_CACHE_DIR", str(Path.home() / ".avp"))) / "adapters"


@dataclass
class UniversalAdapter:
    """A trained universal adapter for a specific model.

    Contains encoder and decoder state dicts plus metadata needed
    to reconstruct the modules and apply affine alignment.
    """

    model_id: str
    """HuggingFace model identifier."""

    model_hash: str
    """SHA-256 of the model config."""

    d_source: int
    """Source model hidden dimension."""

    config: UniversalConfig
    """Universal space configuration used during training."""

    encoder_state_dict: Dict[str, Any]
    """State dict for the encoder nn.Module."""

    decoder_state_dict: Dict[str, Any]
    """State dict for the decoder nn.Module."""

    target_norm: float = 1.0
    """Target model's mean embedding norm for NormMatch."""

    affine_out: Optional[Dict[str, Any]] = None
    """Optional affine transform (W, b) applied after encoder."""

    affine_in: Optional[Dict[str, Any]] = None
    """Optional affine transform (W, b) applied before decoder."""


def _adapter_filename(model_hash: str) -> str:
    """Deterministic filename for a model adapter."""
    return f"{model_hash[:16]}_universal.pt"


def save_adapter(adapter: UniversalAdapter, adapter_dir: Optional[Path] = None) -> Path:
    """Save a UniversalAdapter to disk.

    Args:
        adapter: The trained adapter.
        adapter_dir: Override directory. Defaults to ~/.avp/adapters/.

    Returns:
        Path to the saved file.
    """
    torch = _require_torch()

    save_dir = adapter_dir or _ADAPTER_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = _adapter_filename(adapter.model_hash)
    path = save_dir / filename

    data = {
        "model_id": adapter.model_id,
        "model_hash": adapter.model_hash,
        "d_source": adapter.d_source,
        "config": {
            "d_universal": adapter.config.d_universal,
            "k_tokens": adapter.config.k_tokens,
            "num_layers": adapter.config.num_layers,
            "num_heads": adapter.config.num_heads,
            "dropout": adapter.config.dropout,
            "rollout_steps": adapter.config.rollout_steps,
        },
        "encoder_state_dict": adapter.encoder_state_dict,
        "decoder_state_dict": adapter.decoder_state_dict,
        "target_norm": adapter.target_norm,
        "affine_out": adapter.affine_out,
        "affine_in": adapter.affine_in,
    }
    torch.save(data, path)
    return path


def load_adapter(
    model_hash: str,
    device: str = "cpu",
    adapter_dir: Optional[Path] = None,
) -> Optional[UniversalAdapter]:
    """Load a UniversalAdapter from disk.

    Args:
        model_hash: Model config hash.
        device: Device to load tensors onto.
        adapter_dir: Override directory. Defaults to ~/.avp/adapters/.

    Returns:
        UniversalAdapter or None if not found.
    """
    torch = _require_torch()

    path = find_adapter(model_hash, adapter_dir=adapter_dir)
    if path is None:
        return None

    data = torch.load(path, map_location=device, weights_only=False)

    cfg_dict = data["config"]
    config = UniversalConfig(
        d_universal=cfg_dict["d_universal"],
        k_tokens=cfg_dict["k_tokens"],
        num_layers=cfg_dict["num_layers"],
        num_heads=cfg_dict["num_heads"],
        dropout=cfg_dict["dropout"],
        rollout_steps=cfg_dict.get("rollout_steps", 256),
    )

    return UniversalAdapter(
        model_id=data["model_id"],
        model_hash=data["model_hash"],
        d_source=data["d_source"],
        config=config,
        encoder_state_dict=data["encoder_state_dict"],
        decoder_state_dict=data["decoder_state_dict"],
        target_norm=data.get("target_norm", 1.0),
        affine_out=data.get("affine_out"),
        affine_in=data.get("affine_in"),
    )


def find_adapter(
    model_hash: str,
    adapter_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Check if an adapter file exists for this model.

    Args:
        model_hash: Model config hash.
        adapter_dir: Override directory. Defaults to ~/.avp/adapters/.

    Returns:
        Path to the adapter file, or None if not found.
    """
    search_dir = adapter_dir or _ADAPTER_DIR
    filename = _adapter_filename(model_hash)
    path = search_dir / filename
    return path if path.exists() else None
