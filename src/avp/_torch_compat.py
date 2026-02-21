"""Shared torch compatibility utilities.

Centralizes the lazy torch import pattern used by modules that depend on
torch at runtime but keep it optional at import time.
"""

from .errors import RealignmentError


def require_torch():
    """Import and return the torch module, or raise a clear error.

    Returns:
        The torch module.

    Raises:
        RealignmentError: If torch is not installed.
    """
    try:
        import torch
        return torch
    except ImportError:
        raise RealignmentError(
            "torch is required for this operation. "
            "Install with: pip install avp[latent]"
        )
