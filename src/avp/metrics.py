"""Lightweight observability dataclasses for AVP operations.

Returned by think()/generate() when ``collect_metrics=True``.  No global
registry, no accumulator — callers own the metrics and can log, aggregate,
or discard them as they please.
"""

import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThinkMetrics:
    """Metrics collected during an avp.think() call."""

    model: Optional[str] = None
    """Model name used for thinking."""

    steps: int = 0
    """Number of latent thinking steps requested."""

    has_prior_context: bool = False
    """Whether a prior AVPContext was provided."""

    duration_s: float = 0.0
    """Total wall-clock time for think() in seconds."""

    think_duration_s: float = 0.0
    """Time spent in connector.think()."""


def _PackMetrics(*args, **kwargs):
    """Deprecated. Use ThinkMetrics instead."""
    warnings.warn(
        "PackMetrics is deprecated and will be removed in v0.4. Use ThinkMetrics.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ThinkMetrics(*args, **kwargs)


# Keep PackMetrics importable as an alias
PackMetrics = ThinkMetrics


@dataclass
class UnpackMetrics:
    """Metrics collected during an unpack() call.

    .. deprecated:: 0.3.0
        Use avp.generate() with collect_metrics=True instead.
    """

    input_format: str = "unknown"
    """Detected input format: 'text', 'json', 'binary', or 'packed_message'."""

    has_context: bool = False
    """Whether latent context was available for generation."""

    generated: bool = False
    """Whether model generation was performed (model= was provided)."""

    duration_s: float = 0.0
    """Total wall-clock time for unpack() in seconds."""

    decode_duration_s: float = 0.0
    """Time spent decoding the input."""

    generate_duration_s: float = 0.0
    """Time spent in connector.generate() (if model= provided)."""


@dataclass
class GenerateMetrics:
    """Metrics collected during a generate() call."""

    model: Optional[str] = None
    """Model name used for generation."""

    steps: int = 0
    """Number of latent thinking steps requested."""

    has_prior_context: bool = False
    """Whether prior context was retrieved from the store."""

    stored: bool = False
    """Whether the result was stored in a ContextStore."""

    duration_s: float = 0.0
    """Total wall-clock time for generate() in seconds."""

    think_duration_s: float = 0.0
    """Time spent in connector.think()."""

    generate_duration_s: float = 0.0
    """Time spent in connector.generate()."""


@dataclass
class HandshakeMetrics:
    """Metrics from a handshake resolve() call."""

    resolution_path: str = ""
    """Which rule matched: 'hash_match', 'structural_match',
    'shared_tokenizer', 'avp_map_file', or 'json_fallback'."""

    mode: str = ""
    """Resolved communication mode ('LATENT' or 'JSON')."""

    avp_map_id: str = ""
    """The avp_map_id assigned, if any."""

    duration_s: float = 0.0
    """Wall-clock time for resolve() in seconds."""
