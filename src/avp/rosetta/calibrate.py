"""Build a Rosetta Stone projection map between two models.

Detects vocabulary compatibility (shared tokenizer or BPE overlap)
and returns an AVPMap for zero-shot cross-model projection.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from ..errors import RealignmentError
from ..types import ProjectionMethod
from .._torch_compat import require_torch as _require_torch

logger = logging.getLogger(__name__)


@dataclass
class AVPMap:
    """A vocabulary-mediated projection map between two models.

    The ``w_map`` field holds different data depending on the method:
      - VOCAB_MEDIATED: target model's input embedding weights [vocab_size, D_tgt].
      - VOCAB_OVERLAP: target embeddings for shared tokens [N_shared, D_tgt].
    """

    source_model_id: str
    source_hash: str
    source_dim: int
    target_model_id: str
    target_hash: str
    target_dim: int
    w_map: Any           # torch.Tensor — see class docstring for shape
    bias: Optional[Any]  # torch.Tensor [D_tgt] or None
    target_norm: Any     # torch.Tensor scalar
    method: Union[ProjectionMethod, str]  # ProjectionMethod enum (str accepted for compat)
    anchor_count: int
    validation_score: float
    # Vocab-overlap fields (cross-family projection)
    src_indices: Optional[Any] = None   # LongTensor [N_shared] — source token IDs
    tgt_indices: Optional[Any] = None   # LongTensor [N_shared] — target token IDs
    overlap_count: int = 0
    overlap_ratio: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.method, str):
            self.method = ProjectionMethod(self.method)


def _have_shared_vocab(source_tokenizer: Any, target_tokenizer: Any) -> bool:
    """Check if two tokenizers share the same vocabulary."""
    if not (hasattr(source_tokenizer, "get_vocab") and hasattr(target_tokenizer, "get_vocab")):
        return False
    src_vocab = source_tokenizer.get_vocab()
    tgt_vocab = target_tokenizer.get_vocab()
    return src_vocab == tgt_vocab


def _count_vocab_overlap(
    source_tokenizer: Any,
    target_tokenizer: Any,
    min_overlap: int = 100,
) -> int:
    """Count shared tokens between two tokenizers (lightweight, no torch).

    Used by the handshake to quickly decide if vocab overlap is viable
    without importing torch or building tensors.

    Returns:
        Number of shared tokens, or 0 if below min_overlap.
    """
    if not (hasattr(source_tokenizer, "get_vocab") and hasattr(target_tokenizer, "get_vocab")):
        return 0
    count = len(set(source_tokenizer.get_vocab()) & set(target_tokenizer.get_vocab()))
    return count if count >= min_overlap else 0


def _compute_vocab_overlap(
    source_tokenizer: Any,
    target_tokenizer: Any,
    min_overlap: int = 100,
) -> Any:
    """Find tokens shared between two tokenizers.

    Returns (src_indices, tgt_indices, shared_tokens) sorted by token string,
    or None if overlap count is below min_overlap.
    """
    import numpy as np

    if _count_vocab_overlap(source_tokenizer, target_tokenizer, min_overlap) == 0:
        return None

    src_vocab = source_tokenizer.get_vocab()
    tgt_vocab = target_tokenizer.get_vocab()

    shared_tokens = sorted(set(src_vocab) & set(tgt_vocab))
    src_ids = [src_vocab[t] for t in shared_tokens]
    tgt_ids = [tgt_vocab[t] for t in shared_tokens]

    return (
        np.array(src_ids, dtype=np.intp),
        np.array(tgt_ids, dtype=np.intp),
        shared_tokens,
    )


def _auto_save_map(avp_map: AVPMap) -> None:
    """Save an AVPMap to the registry, logging failures as warnings."""
    from .registry import save_map
    try:
        save_map(avp_map)
        logger.info(
            "Auto-saved AVPMap to registry: %s → %s",
            avp_map.source_hash[:16], avp_map.target_hash[:16],
        )
    except Exception:
        logger.warning("Failed to auto-save AVPMap to registry", exc_info=True)


def calibrate(
    source_model: Any,
    target_model: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    device: Optional[str] = None,
    auto_save: bool = True,
) -> AVPMap:
    """Build a projection map between two models.

    Detects vocabulary compatibility and returns an AVPMap for zero-shot
    cross-model projection. Supports two methods:

    - **vocab_mediated**: When both tokenizers share the same vocabulary
      (same-family models, e.g. Qwen 7B → Qwen 1.5B).
    - **vocab_overlap**: When tokenizers share >=100 BPE tokens
      (cross-family models, e.g. Qwen → Llama, ~85% overlap).

    Args:
        source_model: Source HuggingFace model.
        target_model: Target HuggingFace model.
        source_tokenizer: Source model's tokenizer.
        target_tokenizer: Target model's tokenizer.
        device: Device for computation. Defaults to source model's device.
        auto_save: If True, automatically save the AVPMap to the registry
            (~/.avp/maps/) so subsequent handshakes can discover it. Default True.

    Returns:
        AVPMap with the vocabulary-mediated or vocab-overlap projection.

    Raises:
        ValueError: If models have no compatible projection path (no shared
            tokenizer and <100 overlapping BPE tokens).
    """
    torch = _require_torch()
    from ..handshake import compute_model_hash, extract_model_identity
    from ..realign import compute_target_norm

    if device is None:
        device = str(next(source_model.parameters()).device)

    # Extract identities
    src_identity = extract_model_identity(source_model)
    tgt_identity = extract_model_identity(target_model)
    src_hash = compute_model_hash(source_model.config.to_dict())
    tgt_hash = compute_model_hash(target_model.config.to_dict())

    # Check for shared vocabulary → instant vocab-mediated projection
    shared_vocab = _have_shared_vocab(source_tokenizer, target_tokenizer)

    if not shared_vocab:
        # Check for partial vocabulary overlap → vocab-overlap projection
        overlap_result = _compute_vocab_overlap(source_tokenizer, target_tokenizer)
        if overlap_result is None:
            raise ValueError(
                f"No projection path found between {src_identity.model_id} and "
                f"{tgt_identity.model_id}. Models must share a tokenizer or have "
                f">=100 overlapping BPE tokens for cross-model transfer."
            )

        src_idx_tensor, tgt_idx_tensor, shared_tokens = overlap_result
        src_vocab_size = len(source_tokenizer.get_vocab())
        tgt_vocab_size = len(target_tokenizer.get_vocab())
        overlap_ratio = len(shared_tokens) / min(src_vocab_size, tgt_vocab_size)

        # Pre-index target embeddings for shared tokens
        tgt_input_embeds = target_model.get_input_embeddings()
        if tgt_input_embeds is None or not hasattr(tgt_input_embeds, "weight"):
            raise RealignmentError(
                "Cannot get target input embeddings for vocab-overlap projection."
            )
        tgt_embed_full = tgt_input_embeds.weight.detach().to(torch.float32)
        w_map = tgt_embed_full[tgt_idx_tensor].cpu()  # [N_shared, D_tgt]

        target_norm = compute_target_norm(target_model, device=device)
        src_dim = source_model.config.hidden_size
        tgt_dim = target_model.config.hidden_size
        if src_dim is None or src_dim == 0:
            src_dim = getattr(source_model.config, "n_embd", 0)
        if tgt_dim is None or tgt_dim == 0:
            tgt_dim = getattr(target_model.config, "n_embd", 0)

        logger.info(
            "Vocab overlap: %d shared tokens (%.1f%% of smaller vocab), "
            "src_vocab=%d, tgt_vocab=%d",
            len(shared_tokens), overlap_ratio * 100,
            src_vocab_size, tgt_vocab_size,
        )

        avp_map = AVPMap(
            source_model_id=src_identity.model_id,
            source_hash=src_hash,
            source_dim=src_dim,
            target_model_id=tgt_identity.model_id,
            target_hash=tgt_hash,
            target_dim=tgt_dim,
            w_map=w_map,
            bias=None,
            target_norm=target_norm.cpu(),
            method=ProjectionMethod.VOCAB_OVERLAP,
            anchor_count=0,
            validation_score=overlap_ratio,
            src_indices=src_idx_tensor,
            tgt_indices=tgt_idx_tensor,
            overlap_count=len(shared_tokens),
            overlap_ratio=overlap_ratio,
        )
        if auto_save:
            _auto_save_map(avp_map)
        return avp_map

    # Shared vocabulary → vocab-mediated projection
    tgt_input_embeds = target_model.get_input_embeddings()
    if tgt_input_embeds is None or not hasattr(tgt_input_embeds, "weight"):
        raise RealignmentError(
            "Cannot get target input embeddings for vocab-mediated projection."
        )
    w_map = tgt_input_embeds.weight.detach().to(torch.float32).cpu()
    target_norm = compute_target_norm(target_model, device=device)

    src_dim = source_model.config.hidden_size
    tgt_dim = target_model.config.hidden_size
    if src_dim is None or src_dim == 0:
        src_dim = getattr(source_model.config, "n_embd", 0)
    if tgt_dim is None or tgt_dim == 0:
        tgt_dim = getattr(target_model.config, "n_embd", 0)

    avp_map = AVPMap(
        source_model_id=src_identity.model_id,
        source_hash=src_hash,
        source_dim=src_dim,
        target_model_id=tgt_identity.model_id,
        target_hash=tgt_hash,
        target_dim=tgt_dim,
        w_map=w_map,
        bias=None,
        target_norm=target_norm.cpu(),
        method=ProjectionMethod.VOCAB_MEDIATED,
        anchor_count=0,
        validation_score=1.0,
    )
    if auto_save:
        _auto_save_map(avp_map)
    return avp_map


def calibrate_from_weights(
    source_model_id: str,
    source_config_dict: Dict[str, Any],
    target_model_id: str,
    target_config_dict: Dict[str, Any],
    target_embed_weight: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    auto_save: bool = True,
) -> AVPMap:
    """Create an AVPMap from raw weight tensors without full model loading.

    Used by the vLLM model plugin where source model weights are already
    loaded in the engine and target model embed weights are loaded separately
    from safetensors. Supports vocab-mediated (same tokenizer) and
    vocab-overlap (different tokenizers with shared tokens) methods.

    Args:
        source_model_id: Source model HuggingFace ID.
        source_config_dict: Source model config as dict.
        target_model_id: Target model HuggingFace ID.
        target_config_dict: Target model config as dict.
        target_embed_weight: Target model's embed_tokens weight [vocab, D_tgt].
        source_tokenizer: Source model's tokenizer.
        target_tokenizer: Target model's tokenizer.
        auto_save: If True, save the AVPMap to the registry.

    Returns:
        AVPMap for cross-model projection.
    """
    torch = _require_torch()
    from ..handshake import compute_model_hash

    src_hash = compute_model_hash(source_config_dict)
    tgt_hash = compute_model_hash(target_config_dict)

    src_dim = source_config_dict.get("hidden_size", 0) or source_config_dict.get("n_embd", 0)
    tgt_dim = target_config_dict.get("hidden_size", 0) or target_config_dict.get("n_embd", 0)

    if src_dim <= 0 or tgt_dim <= 0:
        raise ValueError(
            f"Invalid hidden dimensions: source={src_dim}, target={tgt_dim}. "
            "Config dicts must contain 'hidden_size' or 'n_embd'."
        )

    # Compute target norm directly from embed weights
    embed_f32 = target_embed_weight.detach().to(dtype=torch.float32)
    target_norm = embed_f32.norm(dim=1).mean().detach()

    if torch.isnan(target_norm) or target_norm.item() <= 0:
        raise ValueError(
            f"Invalid target_norm ({target_norm.item():.4f}). "
            "Check target_embed_weight for NaN or all-zero rows."
        )

    # Determine projection method from tokenizer compatibility
    shared_vocab = _have_shared_vocab(source_tokenizer, target_tokenizer)

    if shared_vocab:
        avp_map = AVPMap(
            source_model_id=source_model_id,
            source_hash=src_hash,
            source_dim=src_dim,
            target_model_id=target_model_id,
            target_hash=tgt_hash,
            target_dim=tgt_dim,
            w_map=embed_f32.cpu(),
            bias=None,
            target_norm=target_norm.cpu(),
            method=ProjectionMethod.VOCAB_MEDIATED,
            anchor_count=0,
            validation_score=1.0,
        )
    else:
        overlap_result = _compute_vocab_overlap(source_tokenizer, target_tokenizer)
        if overlap_result is None:
            raise ValueError(
                "Insufficient vocabulary overlap (< 100 shared tokens) "
                "between source and target tokenizers."
            )
        src_idx, tgt_idx, shared_tokens = overlap_result
        src_vocab_size = len(source_tokenizer.get_vocab())
        tgt_vocab_size = len(target_tokenizer.get_vocab())
        overlap_ratio = len(shared_tokens) / min(src_vocab_size, tgt_vocab_size)

        w_map = embed_f32[tgt_idx].cpu()

        logger.info(
            "calibrate_from_weights: vocab overlap %d tokens (%.1f%%)",
            len(shared_tokens), overlap_ratio * 100,
        )

        avp_map = AVPMap(
            source_model_id=source_model_id,
            source_hash=src_hash,
            source_dim=src_dim,
            target_model_id=target_model_id,
            target_hash=tgt_hash,
            target_dim=tgt_dim,
            w_map=w_map,
            bias=None,
            target_norm=target_norm.cpu(),
            method=ProjectionMethod.VOCAB_OVERLAP,
            anchor_count=0,
            validation_score=overlap_ratio,
            src_indices=src_idx,
            tgt_indices=tgt_idx,
            overlap_count=len(shared_tokens),
            overlap_ratio=overlap_ratio,
        )

    if auto_save:
        _auto_save_map(avp_map)
    return avp_map
