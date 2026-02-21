"""Runtime cross-model projection via Rosetta Stone maps.

Projects hidden states from a source model's latent space to a target model's
embedding space using a learned linear map (W_map).
"""

from typing import Any, Optional


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        from ..errors import RealignmentError
        raise RealignmentError(
            "torch is required for Rosetta Stone projection. "
            "Install with: pip install avp[latent]"
        )


def apply_cross_model_projection(
    hidden_state: Any,
    w_map: Any,
    target_norm: Any,
    bias: Optional[Any] = None,
) -> Any:
    """Project hidden states from source to target model space.

    Math: h_target = h @ W_map + bias, then normalize to target_norm.
    Same pattern as apply_realignment() in realign.py but supports
    rectangular matrices (D_src -> D_tgt).

    Args:
        hidden_state: Tensor of shape [..., D_src].
        w_map: Projection matrix of shape [D_src, D_tgt].
        target_norm: Scalar target norm from target model embeddings.
        bias: Optional bias vector of shape [D_tgt].

    Returns:
        Projected tensor of shape [..., D_tgt], normalized to target_norm.
    """
    torch = _require_torch()

    original_dtype = hidden_state.dtype
    h = hidden_state.to(torch.float32)
    w = w_map.to(device=h.device, dtype=torch.float32)

    projected = torch.matmul(h, w)

    if bias is not None:
        b = bias.to(device=h.device, dtype=torch.float32)
        projected = projected + b

    # Normalize to target embedding norm
    tn = target_norm.to(device=h.device, dtype=torch.float32)
    norm = projected.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    projected = projected * (tn / norm)

    return projected.to(original_dtype)


def vocabulary_mediated_projection(
    hidden_state: Any,
    source_lm_head_weight: Any,
    target_embed_weight: Any,
    temperature: float = 1.0,
) -> Any:
    """Project hidden states across models via shared vocabulary.

    Cross-model version of project_to_embedding_space() in realign.py.
    Uses the vocabulary as a natural shared coordinate system:
      1. hidden @ W_src.T → logits [vocab_size]
      2. softmax(logits/T) → probability distribution
      3. probs @ W_tgt → target embedding [D_tgt]

    Requires both models to share the same tokenizer (same vocab_size).
    Zero learned parameters — no calibration needed.

    Args:
        hidden_state: Tensor of shape [..., D_src].
        source_lm_head_weight: Source model's output head [vocab_size, D_src].
        target_embed_weight: Target model's input embeddings [vocab_size, D_tgt].
        temperature: Softmax temperature. Lower = sharper (closer to argmax).

    Returns:
        Projected tensor of shape [..., D_tgt].
    """
    torch = _require_torch()

    original_dtype = hidden_state.dtype
    h = hidden_state.to(torch.float32)
    w_src = source_lm_head_weight.detach().to(device=h.device, dtype=torch.float32)
    w_tgt = target_embed_weight.detach().to(device=h.device, dtype=torch.float32)

    # hidden @ W_src^T → logits [..., vocab_size]
    logits = torch.matmul(h, w_src.T)

    # softmax → probability distribution over shared vocabulary
    probs = torch.softmax(logits / temperature, dim=-1)

    # probs @ W_tgt → target embedding [..., D_tgt]
    projected = torch.matmul(probs, w_tgt)

    return projected.to(original_dtype)
