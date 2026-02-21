"""Rosetta Stone map file storage and registry.

Manages .avp-map files on disk — save, load, and find projection maps
for model pairs identified by their config hashes.
"""

import os
from pathlib import Path
from typing import Optional

from ..errors import RealignmentError
from .calibrate import AVPMap


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise RealignmentError(
            "torch is required for Rosetta Stone map I/O. "
            "Install with: pip install avp[latent]"
        )


_MAP_DIR = Path(os.environ.get("AVP_CACHE_DIR", str(Path.home() / ".avp"))) / "maps"


def _map_filename(source_hash: str, target_hash: str) -> str:
    """Deterministic filename for a model pair."""
    return f"{source_hash[:16]}_{target_hash[:16]}.pt"


def save_map(avp_map: AVPMap, map_dir: Optional[Path] = None) -> Path:
    """Save an AVPMap to disk.

    Args:
        avp_map: The calibrated projection map.
        map_dir: Override directory. Defaults to ~/.avp/maps/.

    Returns:
        Path to the saved file.
    """
    torch = _require_torch()

    save_dir = map_dir or _MAP_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = _map_filename(avp_map.source_hash, avp_map.target_hash)
    path = save_dir / filename

    data = {
        "source_model_id": avp_map.source_model_id,
        "source_hash": avp_map.source_hash,
        "source_dim": avp_map.source_dim,
        "target_model_id": avp_map.target_model_id,
        "target_hash": avp_map.target_hash,
        "target_dim": avp_map.target_dim,
        "w_map": avp_map.w_map.cpu(),
        "bias": avp_map.bias.cpu() if avp_map.bias is not None else None,
        "target_norm": avp_map.target_norm.cpu(),
        "method": avp_map.method,
        "anchor_count": avp_map.anchor_count,
        "validation_score": avp_map.validation_score,
    }
    torch.save(data, path)
    return path


def load_map(
    source_hash: str,
    target_hash: str,
    device: str = "cpu",
    map_dir: Optional[Path] = None,
) -> Optional[AVPMap]:
    """Load an AVPMap from disk.

    Args:
        source_hash: Source model config hash.
        target_hash: Target model config hash.
        device: Device to load tensors onto.
        map_dir: Override directory. Defaults to ~/.avp/maps/.

    Returns:
        AVPMap or None if not found.
    """
    torch = _require_torch()

    path = find_map(source_hash, target_hash, map_dir=map_dir)
    if path is None:
        return None

    data = torch.load(path, map_location=device, weights_only=True)
    return AVPMap(
        source_model_id=data["source_model_id"],
        source_hash=data["source_hash"],
        source_dim=data["source_dim"],
        target_model_id=data["target_model_id"],
        target_hash=data["target_hash"],
        target_dim=data["target_dim"],
        w_map=data["w_map"],
        bias=data["bias"],
        target_norm=data["target_norm"],
        method=data["method"],
        anchor_count=data["anchor_count"],
        validation_score=data["validation_score"],
    )


def find_map(
    source_hash: str,
    target_hash: str,
    map_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Check if a map file exists for this model pair.

    Args:
        source_hash: Source model config hash.
        target_hash: Target model config hash.
        map_dir: Override directory. Defaults to ~/.avp/maps/.

    Returns:
        Path to the map file, or None if not found.
    """
    search_dir = map_dir or _MAP_DIR
    filename = _map_filename(source_hash, target_hash)
    path = search_dir / filename
    return path if path.exists() else None


def map_id(source_hash: str, target_hash: str) -> str:
    """Compute the map ID string for a model pair.

    This is the value stored in SessionInfo.avp_map_id and AVPMetadata.avp_map_id.
    """
    return f"{source_hash[:16]}_{target_hash[:16]}"
