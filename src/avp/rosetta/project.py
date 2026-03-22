"""Runtime cross-model projection via Rosetta Stone maps.

Projects hidden states from a source model's latent space to a target model's
embedding space using vocabulary-mediated or vocabulary-overlap projection.

All projection math uses numpy — no torch dependency. Functions accept either
torch tensors or numpy arrays as input (auto-converted via _to_numpy).
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple, Union


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch.Tensor or numpy array to numpy float32."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _to_numpy_idx(x: Any) -> np.ndarray:
    """Convert index tensor/array to numpy integer array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_scalar(x: Any) -> float:
    """Extract float from torch scalar tensor or pass through."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def apply_cross_model_projection(
    hidden_state: Any,
    w_map: Any,
    target_norm: Any,
    bias: Optional[Any] = None,
) -> np.ndarray:
    """Project hidden states from source to target model space.

    Math: h_target = h @ W_map + bias, then normalize to target_norm.

    Args:
        hidden_state: Array-like of shape [..., D_src].
        w_map: Projection matrix of shape [D_src, D_tgt].
        target_norm: Scalar target norm from target model embeddings.
        bias: Optional bias vector of shape [D_tgt].

    Returns:
        Projected numpy array of shape [..., D_tgt], normalized to target_norm.
    """
    h = _to_numpy(hidden_state)
    w = _to_numpy(w_map)
    tn = _to_scalar(target_norm)

    projected = h @ w

    if bias is not None:
        projected = projected + _to_numpy(bias)

    # Normalize to target embedding norm
    norm = np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-6)
    projected = projected * (tn / norm)

    return projected


def vocab_overlap_projection(
    hidden_state: Any,
    source_lm_head_weight: Any,
    shared_target_embed_weight: Any,
    src_indices: Any,
    temperature: float = 1.0,
    target_norm: Optional[Any] = None,
    return_metrics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Project hidden states across models via overlapping vocabulary tokens.

    Cross-family variant of vocabulary_mediated_projection(). Instead of
    requiring identical tokenizers, finds shared tokens between different
    tokenizers and projects through only the overlapping portion:
      1. hidden @ W_src[shared].T → shared logits [N_shared]
      2. softmax(shared_logits/T) → renormalized probabilities
      3. probs @ W_tgt_shared → target embedding [D_tgt]

    Args:
        hidden_state: Array-like of shape [..., D_src].
        source_lm_head_weight: Source model's output head [vocab_size_src, D_src].
        shared_target_embed_weight: Target embeddings for shared tokens [N_shared, D_tgt].
        src_indices: Index array [N_shared] — source token IDs for shared tokens.
        temperature: Softmax temperature. Lower = sharper (closer to argmax).
        target_norm: Optional scalar — mean L2 norm of target model's
            input embeddings. If provided, output is L2-normalized to this value.
        return_metrics: If True, return (projected, metrics_dict).

    Returns:
        Projected numpy array of shape [..., D_tgt]. If return_metrics=True,
        returns (projected, {"entropy": ..., "max_prob": ...}).
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive, got {temperature}. "
            "Use a small value like 0.01 for near-argmax behavior."
        )

    h = _to_numpy(hidden_state)
    w_src = _to_numpy(source_lm_head_weight)
    w_tgt = _to_numpy(shared_target_embed_weight)
    idx = _to_numpy_idx(src_indices)

    # Pre-index source weights to shared tokens, then matmul
    w_src_shared = w_src[idx]  # [N_shared, D_src]
    shared_logits = h @ w_src_shared.T  # [..., N_shared]

    # Renormalized softmax over shared tokens
    probs = _softmax(shared_logits / temperature)

    metrics = None
    if return_metrics:
        log_probs = np.log(np.maximum(probs, 1e-12))
        entropy = -(probs * log_probs).sum(axis=-1)
        max_prob = probs.max(axis=-1)
        # Top-k gap
        if probs.shape[-1] >= 2:
            top2_idx = np.argpartition(probs, -2, axis=-1)[..., -2:]
            top2_vals = np.take_along_axis(probs, top2_idx, axis=-1)
            top1 = top2_vals.max(axis=-1)
            top2 = top2_vals.min(axis=-1)
        else:
            top1 = max_prob
            top2 = np.zeros_like(top1)
        logit_gap = top1 - top2
        h_norm = np.linalg.norm(h, axis=-1)
        metrics = {
            "entropy": entropy,
            "max_prob": max_prob,
            "logit_gap": logit_gap,
            "hidden_state_norm": h_norm,
        }

    # probs @ W_tgt_shared → target embedding [..., D_tgt]
    projected = probs @ w_tgt

    # Normalize to target embedding norm
    if target_norm is not None:
        tn = _to_scalar(target_norm)
        pnorm = np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-6)
        projected = projected * (tn / pnorm)

    if return_metrics:
        # Cosine similarity to nearest target token embedding
        proj_norm = np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-6)
        proj_normalized = projected / proj_norm
        tgt_norm_arr = np.maximum(np.linalg.norm(w_tgt, axis=-1, keepdims=True), 1e-6)
        tgt_normalized = w_tgt / tgt_norm_arr
        cos_sims = proj_normalized @ tgt_normalized.T  # [..., N_shared]
        metrics["nearest_cos_sim"] = cos_sims.max(axis=-1)
        return projected, metrics

    return projected


def vocabulary_mediated_projection(
    hidden_state: Any,
    source_lm_head_weight: Any,
    target_embed_weight: Any,
    temperature: float = 1.0,
    target_norm: Optional[Any] = None,
    return_metrics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Project hidden states across models via shared vocabulary.

    Uses the vocabulary as a natural shared coordinate system:
      1. hidden @ W_src.T → logits [vocab_size]
      2. softmax(logits/T) → probability distribution
      3. probs @ W_tgt → target embedding [D_tgt]

    Requires both models to share the same tokenizer (same vocab_size).
    Zero learned parameters — no calibration needed.

    Args:
        hidden_state: Array-like of shape [..., D_src].
        source_lm_head_weight: Source model's output head [vocab_size, D_src].
        target_embed_weight: Target model's input embeddings [vocab_size, D_tgt].
        temperature: Softmax temperature. Lower = sharper (closer to argmax).
        target_norm: Optional scalar — mean L2 norm of target model's
            input embeddings. If provided, output is L2-normalized to this value.
        return_metrics: If True, return (projected, metrics_dict).

    Returns:
        Projected numpy array of shape [..., D_tgt]. If return_metrics=True,
        returns (projected, {"entropy": ..., "max_prob": ...}).
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive, got {temperature}. "
            "Use a small value like 0.01 for near-argmax behavior."
        )

    h = _to_numpy(hidden_state)
    w_src = _to_numpy(source_lm_head_weight)
    w_tgt = _to_numpy(target_embed_weight)

    # Align vocab dimensions — models in the same family may pad embedding
    # tables differently (e.g. for tensor parallelism). Truncate to the
    # shared prefix which contains all real token embeddings.
    shared_vocab = min(w_src.shape[0], w_tgt.shape[0])
    if w_src.shape[0] != w_tgt.shape[0]:
        w_src = w_src[:shared_vocab]
        w_tgt = w_tgt[:shared_vocab]

    # hidden @ W_src^T → logits [..., vocab_size]
    logits = h @ w_src.T

    # softmax → probability distribution over shared vocabulary
    probs = _softmax(logits / temperature)

    metrics = None
    if return_metrics:
        log_probs = np.log(np.maximum(probs, 1e-12))
        entropy = -(probs * log_probs).sum(axis=-1)
        max_prob = probs.max(axis=-1)
        # Top-k gap
        if probs.shape[-1] >= 2:
            top2_idx = np.argpartition(probs, -2, axis=-1)[..., -2:]
            top2_vals = np.take_along_axis(probs, top2_idx, axis=-1)
            top1 = top2_vals.max(axis=-1)
            top2 = top2_vals.min(axis=-1)
        else:
            top1 = max_prob
            top2 = np.zeros_like(top1)
        logit_gap = top1 - top2
        h_norm = np.linalg.norm(h, axis=-1)
        metrics = {
            "entropy": entropy,
            "max_prob": max_prob,
            "logit_gap": logit_gap,
            "hidden_state_norm": h_norm,
        }

    # probs @ W_tgt → target embedding [..., D_tgt]
    projected = probs @ w_tgt

    # Normalize to target embedding norm
    if target_norm is not None:
        tn = _to_scalar(target_norm)
        pnorm = np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-6)
        projected = projected * (tn / pnorm)

    if return_metrics:
        # Cosine similarity to nearest target token embedding
        proj_norm = np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-6)
        proj_normalized = projected / proj_norm
        tgt_norm_arr = np.maximum(np.linalg.norm(w_tgt, axis=-1, keepdims=True), 1e-6)
        tgt_normalized = w_tgt / tgt_norm_arr
        cos_sims = proj_normalized @ tgt_normalized.T  # [..., vocab_size]
        metrics["nearest_cos_sim"] = cos_sims.max(axis=-1)
        return projected, metrics

    return projected
