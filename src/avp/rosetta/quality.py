"""Per-transfer quality gate for cross-model rosetta projection.

Assesses whether a given transfer is likely to produce useful results
via latent projection, or whether JSON fallback is recommended.

Primary signal: source prompt token count. Single-embedding rosetta works
for short structured prompts (~100-300 tokens, e.g. GSM8K math) but fails
for long comprehension prompts (~1000-2000 tokens, e.g. HotpotQA with
10 paragraphs). Prompt length is a zero-cost proxy for information density.

Secondary signal (opt-in): effective rank ratio of hidden states. High
effective rank means information is spread across many dimensions, which
a single embedding cannot capture. Off by default; included for future
validation.

Usage::

    from avp.rosetta.quality import assess_transfer

    result = assess_transfer(prompt_tokens=len(input_ids[0]))
    if result.recommend_latent:
        # proceed with rosetta projection
        ...
    else:
        # fall back to JSON text transfer
        ...
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TransferQualityConfig:
    """Configuration for the per-transfer quality gate.

    Attributes:
        max_prompt_tokens: Maximum prompt token count for which latent
            transfer is recommended. Default 512 (GSM8K ~100-300 passes,
            HotpotQA ~1000-2000 fails).
        check_effective_rank: Whether to compute effective rank ratio
            of hidden states as a secondary signal. Requires torch and
            a hidden_states tensor. Off by default.
        max_effective_rank_ratio: Threshold for effective rank ratio.
            Above this, JSON fallback is recommended. Only used when
            check_effective_rank=True.
    """

    max_prompt_tokens: int = 512
    check_effective_rank: bool = False
    max_effective_rank_ratio: float = 0.8


@dataclass
class TransferQualityResult:
    """Result of a per-transfer quality assessment.

    Attributes:
        recommend_latent: True if latent transfer is recommended,
            False if JSON fallback is recommended.
        prompt_tokens: The prompt token count that was assessed.
        reason: Human-readable explanation of the recommendation.
        effective_rank_ratio: Effective rank ratio of hidden states,
            or None if not computed.
    """

    recommend_latent: bool
    prompt_tokens: int
    reason: str
    effective_rank_ratio: Optional[float] = None


def _compute_effective_rank_ratio(hidden_states: Any) -> float:
    """Compute effective rank ratio of hidden states via SVD.

    The effective rank (Roy & Vetterli, 2007) measures how spread the
    information is across singular values. A ratio near 1.0 means
    information is uniformly spread (hard to compress into 1 embedding);
    near 0.0 means it's concentrated in few dimensions.

    Args:
        hidden_states: Tensor of shape [seq_len, D] or [1, seq_len, D].

    Returns:
        Ratio in [0, 1] where higher means more spread.
    """
    import torch

    t = hidden_states
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)

    # Squeeze batch dim if present: [1, seq, D] → [seq, D]
    if t.ndim == 3:
        t = t.squeeze(0)

    if t.ndim != 2 or t.shape[0] < 2:
        return 0.0

    # SVD on [seq_len, D]
    s = torch.linalg.svdvals(t.float())

    # Avoid division by zero
    s_sum = s.sum()
    if s_sum == 0:
        return 0.0

    # Normalized singular values → probability distribution
    p = s / s_sum

    # Shannon entropy of the distribution
    log_p = torch.log(p + 1e-12)
    entropy = -(p * log_p).sum().item()

    # Maximum entropy = log(min(seq_len, D))
    max_entropy = float(torch.log(torch.tensor(min(t.shape[0], t.shape[1]), dtype=torch.float32)).item())

    if max_entropy == 0:
        return 0.0

    return entropy / max_entropy


def assess_transfer(
    prompt_tokens: int,
    hidden_states: Any = None,
    config: Optional[TransferQualityConfig] = None,
) -> TransferQualityResult:
    """Assess whether a cross-model transfer should use latent or JSON.

    This is advisory — the caller decides how to act on the result.

    Args:
        prompt_tokens: Number of tokens in the source prompt.
        hidden_states: Optional tensor of hidden states, shape
            [seq_len, D] or [1, seq_len, D]. Only used when
            config.check_effective_rank=True.
        config: Quality gate configuration. Uses defaults if None.

    Returns:
        TransferQualityResult with recommendation and reasoning.
    """
    if config is None:
        config = TransferQualityConfig()

    effective_rank_ratio: Optional[float] = None

    # Primary gate: prompt token count
    if prompt_tokens > config.max_prompt_tokens:
        return TransferQualityResult(
            recommend_latent=False,
            prompt_tokens=prompt_tokens,
            reason=(
                f"prompt_tokens={prompt_tokens} exceeds "
                f"max_prompt_tokens={config.max_prompt_tokens}; "
                f"single embedding unlikely to capture sufficient information"
            ),
            effective_rank_ratio=None,
        )

    # Secondary gate: effective rank (opt-in)
    if config.check_effective_rank and hidden_states is not None:
        effective_rank_ratio = _compute_effective_rank_ratio(hidden_states)
        if effective_rank_ratio > config.max_effective_rank_ratio:
            return TransferQualityResult(
                recommend_latent=False,
                prompt_tokens=prompt_tokens,
                reason=(
                    f"effective_rank_ratio={effective_rank_ratio:.3f} exceeds "
                    f"max={config.max_effective_rank_ratio}; "
                    f"information too spread for single-embedding transfer"
                ),
                effective_rank_ratio=effective_rank_ratio,
            )

    return TransferQualityResult(
        recommend_latent=True,
        prompt_tokens=prompt_tokens,
        reason="transfer within quality thresholds",
        effective_rank_ratio=effective_rank_ratio,
    )
