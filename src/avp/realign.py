"""AVP realignment matrix computation and application.

Ported from LatentMAS models.py:158-213. Maps hidden states from output
space back to input embedding space for same-model latent communication.

Requires torch — this module uses lazy imports so the core SDK works without it.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from .errors import RealignmentError

# Cache directory for realignment matrices
_CACHE_DIR = Path(os.environ.get("AVP_CACHE_DIR", str(Path.home() / ".avp"))) / "realign"


def _require_torch():
    """Lazy import torch, raising a clear error if not available."""
    try:
        import torch
        return torch
    except ImportError:
        raise RealignmentError(
            "torch is required for realignment. Install with: pip install avp[latent]"
        )


def needs_realignment(model_or_config: Any) -> bool:
    """Check whether a model needs realignment (i.e., weights are NOT tied).

    Models with tied input/output embeddings (tie_word_embeddings=True) don't
    need realignment because the embedding spaces are already identical.

    Args:
        model_or_config: HuggingFace model, config, or dict.

    Returns:
        True if realignment is needed (weights are untied).
    """
    if isinstance(model_or_config, dict):
        config_dict = model_or_config
    elif hasattr(model_or_config, "config"):
        config_dict = model_or_config.config.to_dict()
    elif hasattr(model_or_config, "to_dict"):
        config_dict = model_or_config.to_dict()
    else:
        # Default to needing realignment if we can't determine
        return True

    # If tie_word_embeddings is True, no realignment needed
    return not config_dict.get("tie_word_embeddings", False)


def compute_target_norm(model: Any, device: Optional[str] = None) -> Any:
    """Compute target norm from input embeddings.

    This is needed even for tied-weight models: hidden states from the last
    transformer layer have different norms than input embeddings, so we
    always normalize to target_norm before injecting via inputs_embeds.

    Args:
        model: HuggingFace PreTrainedModel.
        device: Target device. Defaults to model's device.

    Returns:
        Scalar tensor: mean L2 norm of input embedding vectors.
    """
    torch = _require_torch()

    input_embeds = (
        model.get_input_embeddings()
        if hasattr(model, "get_input_embeddings")
        else None
    )
    if input_embeds is None or not hasattr(input_embeds, "weight"):
        raise RealignmentError(
            "Cannot compute target norm: input embedding weights not accessible."
        )

    if device is None:
        device = str(next(model.parameters()).device)

    input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
    return input_weight.norm(dim=1).mean().detach()


def compute_realignment_matrix(
    model: Any,
    lambda_reg: float = 1e-5,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Compute the realignment matrix W_realign and target norm.

    Solves: W_realign = (E_out^T E_out + λI)^{-1} E_out^T E_in

    where E_in = input embeddings, E_out = output embeddings (lm_head).

    Args:
        model: HuggingFace PreTrainedModel.
        lambda_reg: Regularization coefficient for the Gram matrix.
        device: Target device. Defaults to model's device.

    Returns:
        Tuple of (W_realign tensor, target_norm scalar tensor).
    """
    torch = _require_torch()

    input_embeds = (
        model.get_input_embeddings()
        if hasattr(model, "get_input_embeddings")
        else None
    )
    output_embeds = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else None
    )
    if output_embeds is None:
        output_embeds = getattr(model, "lm_head", None)

    if (
        input_embeds is None
        or output_embeds is None
        or not hasattr(input_embeds, "weight")
        or not hasattr(output_embeds, "weight")
    ):
        raise RealignmentError(
            "Cannot compute realignment matrix: embedding weights not accessible."
        )

    if device is None:
        device = str(next(model.parameters()).device)

    input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
    output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

    if input_weight.shape[0] != output_weight.shape[0]:
        raise RealignmentError(
            f"Vocab size mismatch: input embeddings have {input_weight.shape[0]} rows, "
            f"output embeddings have {output_weight.shape[0]} rows. "
            f"Call model.resize_token_embeddings() to align them."
        )

    # Gram matrix: E_out^T E_out + λI
    gram = torch.matmul(output_weight.T, output_weight)
    reg = lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    gram = gram + reg

    # RHS: E_out^T E_in
    rhs = torch.matmul(output_weight.T, input_weight)

    # Solve: W_realign = gram^{-1} @ rhs
    realign_matrix = torch.linalg.solve(gram, rhs)

    # Target norm: mean norm of input embeddings
    target_norm = input_weight.norm(dim=1).mean().detach()

    return realign_matrix, target_norm


def normalize_to_target(hidden_state: Any, target_norm: Any) -> Any:
    """Normalize hidden states to target norm.

    This is needed for ALL models (tied and untied). Hidden states from the
    last transformer layer have different norms than input embeddings.
    Following LatentMAS, we always normalize before injecting via inputs_embeds.

    Args:
        hidden_state: Tensor of shape [..., hidden_dim].
        target_norm: Target norm scalar.

    Returns:
        Normalized tensor of same shape and dtype as input.
    """
    torch = _require_torch()

    original_dtype = hidden_state.dtype
    hidden_fp32 = hidden_state.to(torch.float32)

    current_norm = hidden_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = hidden_fp32 * (target_norm / current_norm)

    return normalized.to(original_dtype)


def project_to_embedding_space(
    hidden_state: Any,
    embed_weight: Any,
    temperature: float = 1.0,
) -> Any:
    """Project hidden states to embedding space via softmax soft embedding.

    For tied-weight models, simple normalization doesn't work because hidden
    states from the last transformer layer have very different directional
    structure (cosine similarity ~0.24 to nearest embedding). Instead, we:
    1. Compute logits: hidden @ W^T
    2. Apply softmax to get probability distribution over vocabulary
    3. Compute weighted average of embeddings: probs @ W

    This produces a valid embedding vector (cosine similarity ~1.0 to nearest
    embedding) that the model can understand when injected via inputs_embeds.

    Args:
        hidden_state: Tensor of shape [..., hidden_dim].
        embed_weight: Embedding weight matrix [vocab_size, hidden_dim].
        temperature: Softmax temperature. Lower = sharper (closer to argmax).

    Returns:
        Soft embedding tensor of same shape and dtype as hidden_state.
    """
    torch = _require_torch()

    original_dtype = hidden_state.dtype
    h_fp32 = hidden_state.to(torch.float32)
    w_fp32 = embed_weight.detach().to(torch.float32)

    # hidden @ W^T → logits [..., vocab_size]
    logits = torch.matmul(h_fp32, w_fp32.T)

    # softmax → probability distribution
    probs = torch.softmax(logits / temperature, dim=-1)

    # probs @ W → soft embedding [..., hidden_dim]
    soft_embed = torch.matmul(probs, w_fp32)

    return soft_embed.to(original_dtype)


def apply_realignment(
    hidden_state: Any,
    w_realign: Any,
    target_norm: Any,
) -> Any:
    """Apply realignment to hidden states.

    Projects hidden states from output space to input embedding space,
    then normalizes to match input embedding norm.

    Args:
        hidden_state: Tensor of shape [..., hidden_dim].
        w_realign: Realignment matrix of shape [hidden_dim, hidden_dim].
        target_norm: Target norm scalar.

    Returns:
        Realigned tensor of same shape as input.
    """
    torch = _require_torch()

    original_dtype = hidden_state.dtype
    hidden_fp32 = hidden_state.to(torch.float32)

    aligned = torch.matmul(hidden_fp32, w_realign)

    # Normalize to target norm
    aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / aligned_norm)

    return aligned.to(original_dtype)


def save_realignment_matrix(
    w_realign: Any,
    target_norm: Any,
    model_hash: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Save realignment matrix to disk cache.

    Args:
        w_realign: Realignment matrix tensor.
        target_norm: Target norm scalar tensor.
        model_hash: Model config hash for filename.
        cache_dir: Override cache directory.

    Returns:
        Path to the saved file.
    """
    torch = _require_torch()

    save_dir = cache_dir or _CACHE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / f"{model_hash}.pt"
    torch.save({"w_realign": w_realign.cpu(), "target_norm": target_norm.cpu()}, path)
    return path


def load_realignment_matrix(
    model_hash: str,
    device: str = "cpu",
    cache_dir: Optional[Path] = None,
) -> Optional[Tuple[Any, Any]]:
    """Load realignment matrix from disk cache.

    Args:
        model_hash: Model config hash.
        device: Target device for loaded tensors.
        cache_dir: Override cache directory.

    Returns:
        Tuple of (W_realign, target_norm) or None if not cached.
    """
    torch = _require_torch()

    load_dir = cache_dir or _CACHE_DIR
    path = load_dir / f"{model_hash}.pt"

    if not path.exists():
        return None

    data = torch.load(path, map_location=device, weights_only=True)
    return data["w_realign"], data["target_norm"]


def get_or_compute_realignment(
    model: Any,
    model_hash: str,
    device: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[Any, Any]:
    """Load realignment matrix from cache, or compute and save it.

    Args:
        model: HuggingFace PreTrainedModel.
        model_hash: Model config hash for cache lookup.
        device: Target device.
        cache_dir: Override cache directory.

    Returns:
        Tuple of (W_realign, target_norm).
    """
    target_device = device or str(next(model.parameters()).device)

    # Try loading from cache
    cached = load_realignment_matrix(model_hash, device=target_device, cache_dir=cache_dir)
    if cached is not None:
        return cached

    # Compute and save
    w_realign, target_norm = compute_realignment_matrix(model, device=target_device)
    save_realignment_matrix(w_realign, target_norm, model_hash, cache_dir=cache_dir)

    return w_realign, target_norm
