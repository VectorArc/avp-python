"""Logit-guided decoding for cross-model communication.

Instead of compressing source model information into a single virtual token
(standard rosetta), distributes the signal across the target model's entire
autoregressive generation as additive logit biases.

The source model's vocabulary distribution (from think()) is mapped through
vocabulary overlap to the target model's vocabulary and applied as a constant
bias during generation.
"""

from typing import Any, Optional

from .._torch_compat import require_torch as _require_torch


class CrossModelLogitBias:
    """HuggingFace LogitsProcessor that applies cross-model logit bias.

    During each generation step, adds a scaled bias vector to the target model's
    logits. Implements confidence gating: when the target model is already highly
    confident (max probability > threshold), the bias is suppressed to avoid
    pushing the model away from correct predictions.

    Compatible with transformers LogitsProcessor protocol (__call__ signature).
    """

    def __init__(
        self,
        bias: Any,
        alpha: float = 0.5,
        confidence_threshold: float = 0.8,
    ):
        """Initialize logit bias processor.

        Args:
            bias: Tensor [target_vocab_size] — additive bias for target logits.
                Should be zero-mean over mapped tokens.
            alpha: Scaling factor for the bias. 0.5 = conservative (recommended
                for cross-vocab mapping). Higher = stronger source influence.
            confidence_threshold: When target's max softmax probability exceeds
                this value, the bias is suppressed for that step. Prevents
                "obvious blindness" (biasing away from correct predictions).
        """
        self.bias = bias
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Apply cross-model logit bias with confidence gating.

        Args:
            input_ids: [batch, seq_len] — generated token IDs so far.
            scores: [batch, vocab_size] — target model logits for current step.

        Returns:
            Modified logits [batch, vocab_size].
        """
        torch = _require_torch()

        bias = self.bias.to(device=scores.device, dtype=scores.dtype)

        # Confidence gating: skip bias when target is already confident
        with torch.no_grad():
            probs = torch.softmax(scores, dim=-1)
            max_prob = probs.max(dim=-1).values  # [batch]

        # Per-batch-element mask: 1.0 if uncertain, 0.0 if confident
        mask = (max_prob < self.confidence_threshold).unsqueeze(-1).float()

        return scores + self.alpha * bias * mask


def compute_cross_model_logit_bias(
    source_hidden_state: Any,
    source_lm_head_weight: Any,
    avp_map: Any,
    target_vocab_size: int,
    temperature: float = 1.0,
) -> Any:
    """Compute logit bias vector for cross-model guided decoding.

    Takes the source model's last hidden state (from think()), computes its
    vocabulary distribution, and maps it through vocab overlap to the target
    model's vocabulary as an additive bias.

    Args:
        source_hidden_state: Tensor [1, D_src] or [D_src] — last hidden state
            from source model's think().
        source_lm_head_weight: Source model's lm_head weight [vocab_size_src, D_src].
        avp_map: AVPMap with src_indices, tgt_indices, and method.
        target_vocab_size: Target model's vocabulary size.
        temperature: Softmax temperature for computing source distribution.
            Lower = sharper bias. Default 1.0.

    Returns:
        Tensor [target_vocab_size] — zero-mean additive bias for target logits.
    """
    torch = _require_torch()
    from ..types import ProjectionMethod

    h = source_hidden_state.detach().to(torch.float32)
    if h.dim() == 2:
        h = h.squeeze(0)  # [D_src]

    w_src = source_lm_head_weight.detach().to(device=h.device, dtype=torch.float32)

    # Compute source log-probabilities
    source_logits = torch.matmul(h, w_src.T)  # [vocab_size_src]
    source_log_probs = torch.log_softmax(source_logits / temperature, dim=-1)

    # Initialize target bias to zero (unmapped tokens get no bias)
    target_bias = torch.zeros(target_vocab_size, device=h.device, dtype=torch.float32)

    if avp_map.method == ProjectionMethod.VOCAB_OVERLAP:
        src_idx = avp_map.src_indices.to(h.device)
        tgt_idx = avp_map.tgt_indices.to(h.device)
        target_bias[tgt_idx] = source_log_probs[src_idx]
    elif avp_map.method == ProjectionMethod.VOCAB_MEDIATED:
        # Same tokenizer — direct 1:1 mapping
        shared_vocab = min(source_log_probs.shape[0], target_vocab_size)
        target_bias[:shared_vocab] = source_log_probs[:shared_vocab]
    else:
        # Ridge/Procrustes — no token-level mapping available
        # Fall back to no bias (caller should use rosetta instead)
        return target_bias

    # Zero-mean the mapped entries so the bias doesn't shift the distribution's
    # center of mass (only nudges relative token preferences)
    nonzero_mask = target_bias != 0.0
    if nonzero_mask.any():
        target_bias[nonzero_mask] -= target_bias[nonzero_mask].mean()

    return target_bias
