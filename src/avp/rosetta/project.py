"""Runtime cross-model projection via Rosetta Stone maps.

Projects hidden states from a source model's latent space to a target model's
embedding space using a learned linear map (W_map).
"""

from typing import Any, Dict, Optional, Tuple, Union

from .._torch_compat import require_torch as _require_torch


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


def vocab_overlap_projection(
    hidden_state: Any,
    source_lm_head_weight: Any,
    shared_target_embed_weight: Any,
    src_indices: Any,
    temperature: float = 1.0,
    return_metrics: bool = False,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """Project hidden states across models via overlapping vocabulary tokens.

    Cross-family variant of vocabulary_mediated_projection(). Instead of
    requiring identical tokenizers, finds shared tokens between different
    tokenizers and projects through only the overlapping portion:
      1. hidden @ W_src.T → full logits [vocab_size_src]
      2. full_logits[..., src_indices] → shared logits [N_shared]
      3. softmax(shared_logits/T) → renormalized probabilities
      4. probs @ W_tgt_shared → target embedding [D_tgt]

    Args:
        hidden_state: Tensor of shape [..., D_src].
        source_lm_head_weight: Source model's output head [vocab_size_src, D_src].
        shared_target_embed_weight: Target embeddings for shared tokens [N_shared, D_tgt].
        src_indices: LongTensor [N_shared] — source token IDs for shared tokens.
        temperature: Softmax temperature. Lower = sharper (closer to argmax).
        return_metrics: If True, return (projected, metrics_dict) with entropy
            and max_prob of the softmax distribution.

    Returns:
        Projected tensor of shape [..., D_tgt]. If return_metrics=True,
        returns (projected, {"entropy": ..., "max_prob": ...}).
    """
    torch = _require_torch()

    original_dtype = hidden_state.dtype
    h = hidden_state.to(torch.float32)
    w_src = source_lm_head_weight.detach().to(device=h.device, dtype=torch.float32)
    w_tgt = shared_target_embed_weight.detach().to(device=h.device, dtype=torch.float32)
    idx = src_indices.to(device=h.device)

    # Pre-index source weights to shared tokens, then matmul
    # Equivalent to: (h @ W_src.T)[..., idx], but avoids computing
    # logits for non-shared tokens (~15% savings at 85% overlap,
    # more for lower-overlap pairs).
    w_src_shared = w_src[idx]  # [N_shared, D_src]
    shared_logits = torch.matmul(h, w_src_shared.T)  # [..., N_shared]

    # Renormalized softmax over shared tokens
    probs = torch.softmax(shared_logits / temperature, dim=-1)

    if return_metrics:
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * log_probs).sum(dim=-1)
        max_prob = probs.max(dim=-1).values
        # Top-k gap: difference between top-1 and top-2 probability
        topk = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
        top1 = topk.values[..., 0]
        top2 = topk.values[..., 1] if topk.values.shape[-1] > 1 else torch.zeros_like(top1)
        logit_gap = top1 - top2
        # Hidden state norm (before projection)
        h_norm = h.norm(dim=-1)

    # probs @ W_tgt_shared → target embedding [..., D_tgt]
    projected = torch.matmul(probs, w_tgt)

    if return_metrics:
        # Cosine similarity to nearest target token embedding
        proj_f32 = projected.to(torch.float32)
        proj_normalized = proj_f32 / proj_f32.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tgt_normalized = w_tgt / w_tgt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos_sims = torch.matmul(proj_normalized, tgt_normalized.T)  # [..., N_shared]
        nearest_cos_sim = cos_sims.max(dim=-1).values

        metrics = {
            "entropy": entropy,
            "max_prob": max_prob,
            "logit_gap": logit_gap,
            "hidden_state_norm": h_norm,
            "nearest_cos_sim": nearest_cos_sim,
        }
        return projected.to(original_dtype), metrics
    return projected.to(original_dtype)


def vocabulary_mediated_projection(
    hidden_state: Any,
    source_lm_head_weight: Any,
    target_embed_weight: Any,
    temperature: float = 1.0,
    return_metrics: bool = False,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
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
        return_metrics: If True, return (projected, metrics_dict) with entropy
            and max_prob of the softmax distribution.

    Returns:
        Projected tensor of shape [..., D_tgt]. If return_metrics=True,
        returns (projected, {"entropy": ..., "max_prob": ...}).
    """
    torch = _require_torch()

    original_dtype = hidden_state.dtype
    h = hidden_state.to(torch.float32)
    w_src = source_lm_head_weight.detach().to(device=h.device, dtype=torch.float32)
    w_tgt = target_embed_weight.detach().to(device=h.device, dtype=torch.float32)

    # Align vocab dimensions — models in the same family may pad embedding
    # tables differently (e.g. for tensor parallelism). Truncate to the
    # shared prefix which contains all real token embeddings.
    shared_vocab = min(w_src.shape[0], w_tgt.shape[0])
    if w_src.shape[0] != w_tgt.shape[0]:
        w_src = w_src[:shared_vocab]
        w_tgt = w_tgt[:shared_vocab]

    # hidden @ W_src^T → logits [..., vocab_size]
    logits = torch.matmul(h, w_src.T)

    # softmax → probability distribution over shared vocabulary
    probs = torch.softmax(logits / temperature, dim=-1)

    if return_metrics:
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * log_probs).sum(dim=-1)
        max_prob = probs.max(dim=-1).values
        # Top-k gap: difference between top-1 and top-2 probability
        topk = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
        top1 = topk.values[..., 0]
        top2 = topk.values[..., 1] if topk.values.shape[-1] > 1 else torch.zeros_like(top1)
        logit_gap = top1 - top2
        # Hidden state norm (before projection)
        h_norm = h.norm(dim=-1)

    # probs @ W_tgt → target embedding [..., D_tgt]
    projected = torch.matmul(probs, w_tgt)

    if return_metrics:
        # Cosine similarity to nearest target token embedding
        proj_f32 = projected.to(torch.float32)
        proj_normalized = proj_f32 / proj_f32.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tgt_normalized = w_tgt / w_tgt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos_sims = torch.matmul(proj_normalized, tgt_normalized.T)  # [..., vocab_size]
        nearest_cos_sim = cos_sims.max(dim=-1).values

        metrics = {
            "entropy": entropy,
            "max_prob": max_prob,
            "logit_gap": logit_gap,
            "hidden_state_norm": h_norm,
            "nearest_cos_sim": nearest_cos_sim,
        }
        return projected.to(original_dtype), metrics
    return projected.to(original_dtype)
