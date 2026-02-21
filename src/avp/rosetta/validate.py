"""Validate cross-model projection quality at session start.

Two-tier validation:
1. Cosine similarity (fast gate, ~1ms): project source hidden states, compare
   to target embeddings. If cos_sim < threshold, skip perplexity and return JSON.
2. Pseudo-perplexity (~30ms): feed projected embeddings as inputs_embeds into
   target model, compute cross-entropy against known next tokens. Only available
   when both models share the same tokenizer.

Returns a recommended CommunicationMode (LATENT / HYBRID / JSON).
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from .._torch_compat import require_torch as _require_torch
from ..types import CommunicationMode, ProjectionMethod
from .calibrate import AVPMap

# Five test sentences — one from each category in DEFAULT_ANCHORS.
VALIDATION_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Explain the difference between a stack and a queue in computer science.",
    "If x plus three equals seven, then x equals four.",
    "def hello_world(): print('Hello, world!')",
    "Bonjour, comment allez-vous aujourd'hui?",
]


@dataclass
class ValidationConfig:
    """Thresholds for projection quality validation."""

    cosine_sim_threshold: float = 0.5  # Below → instant JSON (skip perplexity)
    perplexity_latent: float = 20.0    # Below → LATENT
    perplexity_json: float = 100.0     # Above → JSON, between → HYBRID
    num_test_texts: int = 5


@dataclass
class ValidationResult:
    """Result of projection quality validation."""

    cosine_similarity: float                   # Mean cos sim across test vectors
    perplexity: Optional[float]                # None if shared tokenizer not available
    recommended_mode: CommunicationMode
    detail: str                                # Human-readable explanation


def _have_shared_vocab(source_tokenizer: Any, target_tokenizer: Any) -> bool:
    """Check if two tokenizers share the same vocabulary."""
    if not (hasattr(source_tokenizer, "get_vocab") and hasattr(target_tokenizer, "get_vocab")):
        return False
    return source_tokenizer.get_vocab() == target_tokenizer.get_vocab()


def _extract_per_token_hidden_states(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: str,
) -> Any:
    """Run texts through model and collect per-token last-layer hidden states.

    Returns a list of tensors, each of shape [seq_len, hidden_dim].
    """
    torch = _require_torch()

    all_hidden = []
    for text in texts:
        encoded = tokenizer(
            text, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=128,
        )
        input_ids = encoded["input_ids"].to(device)
        if "attention_mask" in encoded:
            attention_mask = encoded["attention_mask"].to(device)
        else:
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Last hidden layer, all tokens: [seq_len, hidden_dim]
        hidden = outputs.hidden_states[-1][0].to(torch.float32)
        all_hidden.append((hidden, input_ids[0]))

    return all_hidden


def _compute_cosine_similarity(
    projected: Any,
    target_embeds: Any,
    token_ids: Any,
    is_next_token: bool = False,
) -> float:
    """Compute mean cosine similarity between projected vectors and target embeddings.

    Args:
        projected: Projected hidden states [seq_len, D_tgt].
        target_embeds: Target input embedding weight matrix [vocab_size, D_tgt].
        token_ids: Token IDs [seq_len].
        is_next_token: If True, projected[i] predicts token_ids[i+1] (vocab-mediated).
            If False, projected[i] corresponds to token_ids[i] (learned maps).

    Returns:
        Mean cosine similarity across compared positions.
    """
    torch = _require_torch()

    if is_next_token:
        # Vocab-mediated: hidden[i] → lm_head → softmax → next-token embedding.
        # Compare projected[i] to target_embed[token_ids[i+1]].
        if projected.shape[0] < 2:
            return 0.0
        proj = projected[:-1]           # [seq_len-1, D_tgt]
        target_vecs = target_embeds[token_ids[1:]].to(
            dtype=torch.float32, device=proj.device
        )
    else:
        # Learned maps: projected[i] maps to same-position embedding.
        proj = projected
        target_vecs = target_embeds[token_ids].to(
            dtype=torch.float32, device=proj.device
        )

    # Normalize and compute cosine similarity
    proj_norm = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt_norm = target_vecs / target_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    cos_sims = (proj_norm * tgt_norm).sum(dim=-1)
    return cos_sims.mean().item()


def _compute_pseudo_perplexity(
    model: Any,
    projected: Any,
    token_ids: Any,
    device: str,
) -> float:
    """Feed projected embeddings into target model and measure prediction quality.

    Computes cross-entropy of logits[i] vs token_ids[i+1] for all positions,
    then perplexity = exp(mean_loss).

    Args:
        model: Target model.
        projected: Projected hidden states [seq_len, D_tgt].
        token_ids: Token IDs [seq_len] from the shared tokenizer.
        device: Computation device.

    Returns:
        Pseudo-perplexity score.
    """
    torch = _require_torch()

    if projected.shape[0] < 2:
        return float("inf")

    # Match the model's parameter dtype (e.g. bfloat16) to avoid mat-mul errors
    model_dtype = next(model.parameters()).dtype
    inputs_embeds = projected.unsqueeze(0).to(device=device, dtype=model_dtype)

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )

    # logits: [1, seq_len, vocab_size]
    logits = outputs.logits[0].to(torch.float32)  # [seq_len, vocab_size]

    # Shift: predict token[i+1] from logits[i]
    shift_logits = logits[:-1]          # [seq_len-1, vocab_size]
    shift_labels = token_ids[1:].to(device)  # [seq_len-1]

    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
    perplexity = torch.exp(loss).item()

    return perplexity


def _apply_projection(
    hidden_states: Any,
    avp_map: AVPMap,
    source_model: Any,
) -> Any:
    """Apply the AVP map projection to hidden states.

    Uses the same code path as runtime: vocabulary_mediated_projection for
    vocab-mediated maps, apply_cross_model_projection for learned maps.
    """
    if avp_map.method == ProjectionMethod.VOCAB_MEDIATED:
        from .project import vocabulary_mediated_projection
        source_lm_head = source_model.get_output_embeddings()
        if source_lm_head is None:
            source_lm_head = getattr(source_model, "lm_head", None)
        return vocabulary_mediated_projection(
            hidden_states,
            source_lm_head_weight=source_lm_head.weight,
            target_embed_weight=avp_map.w_map,
        )
    else:
        from .project import apply_cross_model_projection
        return apply_cross_model_projection(
            hidden_states, avp_map.w_map, avp_map.target_norm, avp_map.bias
        )


def validate_projection(
    source_model: Any,
    target_model: Any,
    avp_map: AVPMap,
    source_tokenizer: Any,
    target_tokenizer: Any,
    config: Optional[ValidationConfig] = None,
    device: Optional[str] = None,
) -> ValidationResult:
    """Validate cross-model projection quality.

    Runs test sentences through the source model, projects hidden states
    via the AVP map, and measures quality against the target model.

    Args:
        source_model: Source HuggingFace model.
        target_model: Target HuggingFace model.
        avp_map: AVPMap (vocab_mediated, ridge, or procrustes).
        source_tokenizer: Source model's tokenizer.
        target_tokenizer: Target model's tokenizer.
        config: Validation thresholds. Defaults to ValidationConfig().
        device: Computation device. Defaults to source model's device.

    Returns:
        ValidationResult with cosine_similarity, perplexity, recommended_mode,
        and a human-readable detail string.
    """
    torch = _require_torch()

    if config is None:
        config = ValidationConfig()
    if device is None:
        device = str(next(source_model.parameters()).device)

    texts = VALIDATION_TEXTS[: config.num_test_texts]

    # Get target input embedding weights for cosine similarity comparison
    tgt_input_embeds = target_model.get_input_embeddings()
    tgt_embed_weight = tgt_input_embeds.weight.detach().to(torch.float32)

    # Extract per-token hidden states from source model
    source_data = _extract_per_token_hidden_states(
        source_model, source_tokenizer, texts, device,
    )

    # Vocab-mediated projection goes through lm_head → softmax, which predicts
    # the next token. Cosine similarity must compare projected[i] to the
    # embedding of token_ids[i+1], not token_ids[i].
    is_next_token = (avp_map.method == ProjectionMethod.VOCAB_MEDIATED)

    # Project and compute cosine similarity
    all_cos_sims = []
    projected_list = []
    token_ids_list = []

    for hidden_states, token_ids in source_data:
        projected = _apply_projection(hidden_states, avp_map, source_model)
        projected_list.append(projected)
        token_ids_list.append(token_ids)

        cos_sim = _compute_cosine_similarity(
            projected, tgt_embed_weight, token_ids, is_next_token=is_next_token,
        )
        all_cos_sims.append(cos_sim)

    mean_cos_sim = sum(all_cos_sims) / len(all_cos_sims)

    # Fast gate: low cosine similarity → instant JSON
    if mean_cos_sim < config.cosine_sim_threshold:
        return ValidationResult(
            cosine_similarity=mean_cos_sim,
            perplexity=None,
            recommended_mode=CommunicationMode.JSON,
            detail=(
                f"cos_sim={mean_cos_sim:.3f} < {config.cosine_sim_threshold} "
                f"threshold — projection unreliable, using JSON"
            ),
        )

    # Pseudo-perplexity (only if shared tokenizer)
    shared_vocab = _have_shared_vocab(source_tokenizer, target_tokenizer)
    if not shared_vocab:
        # Can't compute perplexity without shared tokenizer — use cos sim only
        if mean_cos_sim > 0.8:
            mode = CommunicationMode.LATENT
            detail = (
                f"cos_sim={mean_cos_sim:.3f} (high) — no shared tokenizer "
                f"for perplexity check, recommending LATENT"
            )
        else:
            mode = CommunicationMode.HYBRID
            detail = (
                f"cos_sim={mean_cos_sim:.3f} (moderate) — no shared tokenizer "
                f"for perplexity check, recommending HYBRID"
            )
        return ValidationResult(
            cosine_similarity=mean_cos_sim,
            perplexity=None,
            recommended_mode=mode,
            detail=detail,
        )

    # Compute pseudo-perplexity across all test sentences
    all_ppl = []
    for projected, token_ids in zip(projected_list, token_ids_list):
        ppl = _compute_pseudo_perplexity(target_model, projected, token_ids, device)
        all_ppl.append(ppl)

    mean_ppl = sum(all_ppl) / len(all_ppl)

    # Threshold decision
    if mean_ppl < config.perplexity_latent:
        mode = CommunicationMode.LATENT
        detail = (
            f"cos_sim={mean_cos_sim:.3f}, perplexity={mean_ppl:.1f} "
            f"< {config.perplexity_latent} — high-quality projection"
        )
    elif mean_ppl > config.perplexity_json:
        mode = CommunicationMode.JSON
        detail = (
            f"cos_sim={mean_cos_sim:.3f}, perplexity={mean_ppl:.1f} "
            f"> {config.perplexity_json} — projection unreliable, using JSON"
        )
    else:
        mode = CommunicationMode.HYBRID
        detail = (
            f"cos_sim={mean_cos_sim:.3f}, perplexity={mean_ppl:.1f} "
            f"(between {config.perplexity_latent} and {config.perplexity_json}) "
            f"— meaningful but lossy, using HYBRID"
        )

    return ValidationResult(
        cosine_similarity=mean_cos_sim,
        perplexity=mean_ppl,
        recommended_mode=mode,
        detail=detail,
    )
