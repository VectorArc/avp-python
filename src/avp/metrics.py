"""Lightweight observability dataclasses for AVP operations.

Returned by think()/generate() when ``collect_metrics=True``.  No global
registry, no accumulator — callers own the metrics and can log, aggregate,
or discard them as they please.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DebugConfig:
    """Configuration for AVP debug observability.

    Controls which diagnostics are collected during ``avp.think()`` and
    ``avp.generate()`` calls.  Pass as the ``debug_config`` parameter::

        debug_config=DebugConfig()                # health diagnostics
        debug_config=DebugConfig(compare=True)    # health + text baseline
        debug_config=DebugConfig(step_tokens=5)   # health + per-step tokens

    Attributes:
        health: Collect transfer health diagnostics — NaN/Inf detection,
            norm trajectory, projection metrics, quality gate result.
            Always True; the base tier of debug observability.
        compare: Generate a text-only baseline and compute word overlap
            with the latent output.  Roughly doubles generation time.
        step_tokens: Number of top-k tokens to decode at each latent
            step via lm_head projection.  Shows what the model would
            generate if forced to speak at each thinking step.
            Set to 0 to disable.  **Not yet implemented.**
        step_snapshots: Generate a text snapshot every N latent steps
            from the current KV-cache (read-only).  Set to 0 to disable.
            **Not yet implemented.**
    """

    health: bool = True
    compare: bool = False
    step_tokens: int = 0
    step_snapshots: int = 0



@dataclass
class TransferDiagnostics:
    """Debug diagnostics for latent transfers.

    Created when ``debug=True`` is passed to ``avp.think()`` or
    ``avp.generate()``.  Critical issues (empty output, NaN/Inf) always
    emit ``RuntimeWarning`` regardless of the debug flag; these fields
    are only populated when diagnostics are collected.
    """

    # Always-on health checks
    output_empty: bool = False
    has_nan: bool = False
    has_inf: bool = False
    output_length: int = 0

    # Context metadata
    transfer_mode: str = ""
    source_model: Optional[str] = None
    target_model: Optional[str] = None
    prompt_tokens: int = 0

    # Quality gate (rosetta)
    quality_gate_passed: Optional[bool] = None
    quality_gate_reason: str = ""

    # Projection metrics (debug=True, rosetta only)
    projection_method: str = ""
    hidden_state_norm: Optional[float] = None
    nearest_cos_sim: Optional[float] = None

    # Norm trajectory (debug=True, latent only)
    norm_trajectory: Optional[List[float]] = None

    # Comparison mode (debug="compare")
    text_baseline_output: Optional[str] = None
    text_overlap: Optional[float] = None

    # Collected warnings
    warnings: List[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """True if no red flags detected."""
        return (
            not self.output_empty
            and not self.has_nan
            and not self.has_inf
        )

    def summary(self) -> str:
        """One-line status string."""
        parts = []
        if not self.healthy:
            flags = []
            if self.output_empty:
                flags.append("EMPTY_OUTPUT")
            if self.has_nan:
                flags.append("NaN")
            if self.has_inf:
                flags.append("Inf")
            parts.append("UNHEALTHY[" + ",".join(flags) + "]")
        else:
            parts.append("OK")

        if self.transfer_mode:
            parts.append(f"mode={self.transfer_mode}")
        if self.projection_method:
            parts.append(f"projection={self.projection_method}")
        if self.nearest_cos_sim is not None:
            parts.append(f"cos_sim={self.nearest_cos_sim:.3f}")
        if self.norm_trajectory:
            parts.append(f"norms=[{self.norm_trajectory[0]:.1f}..{self.norm_trajectory[-1]:.1f}]")
        if self.text_overlap is not None:
            parts.append(f"overlap={self.text_overlap:.0%}")
        if self.warnings:
            parts.append(f"warnings={len(self.warnings)}")

        return " | ".join(parts)


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

    diagnostics: Optional[TransferDiagnostics] = None
    """Debug diagnostics (populated when debug=True)."""


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

    diagnostics: Optional[TransferDiagnostics] = None
    """Debug diagnostics (populated when debug=True)."""


