"""Rosetta Stone — cross-model latent projection for AVP.

Calibrates and applies linear maps between different models' hidden spaces,
enabling latent communication across model architectures.
"""

from .calibrate import AVPMap, DEFAULT_ANCHORS, calibrate
from .project import apply_cross_model_projection, vocabulary_mediated_projection
from .registry import find_map, load_map, map_id, save_map
from .validate import ValidationConfig, ValidationResult, validate_projection

__all__ = [
    # Calibration
    "AVPMap",
    "DEFAULT_ANCHORS",
    "calibrate",
    # Projection
    "apply_cross_model_projection",
    "vocabulary_mediated_projection",
    # Registry
    "save_map",
    "load_map",
    "find_map",
    "map_id",
    # Validation
    "ValidationConfig",
    "ValidationResult",
    "validate_projection",
]
