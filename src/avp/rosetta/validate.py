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
    perplexity_latent: float = 50.0    # Below → LATENT (calibrated: Qwen2.5-1.5B→0.5B ppl=25.8 works)
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
    projected_last_hidden: Any,
    token_ids: Any,
    device: str,
) -> float:
    """Measure if the target model can function after receiving a projected embedding.

    Mirrors how the pipeline uses projections: inject ONE projected embedding
    (the source model's last-token hidden state) to prime the target model's
    context, then feed the actual text tokens and measure prediction quality.

    Perplexity = exp(mean cross-entropy) on the text portion only.

    Args:
        model: Target model.
        projected_last_hidden: Projected last-token hidden state [D_tgt].
        token_ids: Token IDs [seq_len] from the shared tokenizer.
        device: Computation device.

    Returns:
        Pseudo-perplexity score.
    """
    torch = _require_torch()

    if token_ids.shape[0] < 2:
        return float("inf")

    model_dtype = next(model.parameters()).dtype

    # Get target model's own embeddings for the text tokens
    tgt_input_embeds = model.get_input_embeddings()
    text_embeds = tgt_input_embeds(token_ids.to(device))  # [seq_len, D_tgt]

    # Prepend the single projected embedding as a context token:
    # [projected_embed, text_embed[0], text_embed[1], ..., text_embed[N-1]]
    proj_embed = projected_last_hidden.unsqueeze(0).to(
        device=device, dtype=model_dtype,
    )  # [1, D_tgt]
    combined = torch.cat([proj_embed, text_embeds], dim=0)  # [1+seq_len, D_tgt]
    combined = combined.unsqueeze(0).to(model_dtype)  # [1, 1+seq_len, D_tgt]

    with torch.no_grad():
        outputs = model(
            inputs_embeds=combined,
            return_dict=True,
        )

    # logits: [1, 1+seq_len, vocab_size]
    logits = outputs.logits[0].to(torch.float32)  # [1+seq_len, vocab_size]

    # Score only the text portion: logits[i] predicts the token after position i.
    # Position 0 = projected embed → logits[0] predicts token_ids[0]
    # Position 1 = text_embed[0]   → logits[1] predicts token_ids[1]
    # ...
    # Position N = text_embed[N-1] → logits[N] predicts token_ids[N] (doesn't exist)
    # So: score logits[0..N-1] against token_ids[0..N-1]
    score_logits = logits[:token_ids.shape[0]]        # [seq_len, vocab_size]
    score_labels = token_ids.to(device)               # [seq_len]

    loss = torch.nn.functional.cross_entropy(score_logits, score_labels)
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

    # Compute pseudo-perplexity across all test sentences.
    # Use the last-token projected hidden state (matching how the pipeline
    # injects a single embedding to prime the target model's KV-cache).
    all_ppl = []
    for projected, token_ids in zip(projected_list, token_ids_list):
        last_projected = projected[-1]  # [D_tgt] — last token's projection
        ppl = _compute_pseudo_perplexity(
            target_model, last_projected, token_ids, device,
        )
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
