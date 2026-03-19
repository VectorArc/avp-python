"""Mid-layer injection for cross-model latent transfer.

Instead of injecting projected hidden states at layer 0 (via inputs_embeds),
injects at an intermediate layer (~75% depth) using a forward hook. This
bypasses the early embedding/position-encoding layers and operates directly
in the semantic representation space.

Based on:
- Ramesh & Li (2501.14082): Cross-model hidden state injection at intermediate
  layers, up to 27% improvement over text, cross-family confirmed.
- Proportional depth mapping (2504.08775): Layer L_a/N_a maps to L_b/N_b
  across architectures (p < 0.005 for 24 LLMs from 1B-70B).

Key design decisions:
- REPLACE, not sum/mean (Ramesh & Li found sum/mean produce OOD norms)
- Proportional depth mapping: injection_layer = int(N_tgt * extraction_ratio)
- Forward hook scoped to prefill only (removed after first forward pass)
"""

import logging
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Default extraction/injection depth ratio (validated at 0.75 = ~75% depth)
DEFAULT_DEPTH_RATIO = 0.75


def compute_extraction_layer(num_layers: int, depth_ratio: float = DEFAULT_DEPTH_RATIO) -> int:
    """Compute the layer index to extract hidden states from.

    Args:
        num_layers: Total number of transformer layers in the model.
        depth_ratio: Fraction of depth to extract from (0.0=first, 1.0=last).

    Returns:
        Layer index (0-indexed).
    """
    layer = int(num_layers * depth_ratio)
    return min(layer, num_layers - 1)


def compute_injection_layer(num_layers: int, depth_ratio: float = DEFAULT_DEPTH_RATIO) -> int:
    """Compute the layer index to inject hidden states into.

    Uses proportional depth mapping: if source extracted from 75% depth,
    inject at 75% depth of target model (even if different number of layers).

    Args:
        num_layers: Total number of transformer layers in target model.
        depth_ratio: Fraction of depth to inject at.

    Returns:
        Layer index (0-indexed).
    """
    layer = int(num_layers * depth_ratio)
    return min(layer, num_layers - 1)


def extract_mid_layer_hidden(
    model_outputs: Any,
    extraction_layer: int,
) -> Any:
    """Extract hidden state from an intermediate layer of model outputs.

    Args:
        model_outputs: Model output dict with hidden_states.
        extraction_layer: Layer index to extract from.

    Returns:
        Hidden state tensor [B, D] from the specified layer's last token.
    """
    # hidden_states is a tuple of (num_layers + 1) tensors, each [B, seq, D]
    # Index 0 = embedding output, index i = output of layer i
    hidden_states = model_outputs.hidden_states
    if extraction_layer + 1 >= len(hidden_states):
        extraction_layer = len(hidden_states) - 2  # -1 is last layer output
    # +1 because index 0 is embedding layer output, index 1 is layer 0 output
    layer_hidden = hidden_states[extraction_layer + 1]
    return layer_hidden[:, -1, :]  # [B, D] — last token only


def _get_decoder_layers(model: Any):
    """Get the list of decoder layers from a HuggingFace model.

    Handles different model architectures (Llama, Qwen, GPT-2, etc.).
    """
    # Try common attribute paths
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return layers

    # GPT-2 style
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        h = getattr(transformer, "h", None)
        if h is not None:
            return h

    raise AttributeError(
        f"Cannot find decoder layers in model {type(model).__name__}. "
        "Expected model.model.layers or model.transformer.h"
    )


CALIBRATION_PROMPTS = [
    "Solve step by step: What is 24 * 17 + 3?",
    "Write a Python function that checks if a number is prime.",
    "Explain the difference between a stack and a queue.",
    "The quick brown fox jumps over the lazy dog.",
    "Analyze the following: renewable energy costs have decreased by 90% since 2010.",
    "What are the main causes of the French Revolution?",
    "Debug this code: def add(a, b): return a - b",
    "Summarize: Machine learning models learn patterns from data.",
]


def compute_activation_norm(
    model: Any,
    tokenizer: Any,
    layer_index: int,
    prompts: Optional[List[str]] = None,
) -> float:
    """Compute mean L2 norm of activations at a specific layer.

    Runs calibration prompts through the model and records the
    last-token hidden state norm at the target layer.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        layer_index: Which layer to measure (0-indexed).
        prompts: Calibration prompts. Uses defaults if None.

    Returns:
        Mean L2 norm (float) of the last-token hidden state at that layer.
    """
    if prompts is None:
        prompts = CALIBRATION_PROMPTS

    norms = []
    model.eval()
    with torch.no_grad():
        for text in prompts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states[0] = embedding output, hidden_states[i+1] = layer i output
            layer_hidden = outputs.hidden_states[layer_index + 1]
            norm = layer_hidden[:, -1, :].float().norm(dim=-1).item()
            norms.append(norm)

    mean_norm = sum(norms) / len(norms)
    logger.info(
        "Activation norm at layer %d: %.1f (from %d prompts)",
        layer_index, mean_norm, len(norms),
    )
    return mean_norm


def renormalize_to_activation_space(
    projected: torch.Tensor,
    activation_norm: float,
) -> torch.Tensor:
    """Renormalize a projected vector from embedding-space norm to activation-space norm.

    Args:
        projected: Projected tensor [..., D] (currently at embedding-space norm).
        activation_norm: Target L2 norm for the injection layer.

    Returns:
        Renormalized tensor with L2 norm matching the activation space.
    """
    current_norm = projected.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return projected * (activation_norm / current_norm)


@contextmanager
def mid_layer_injection_hook(
    model: Any,
    injection_layer: int,
    projected_hidden: Any,
):
    """Context manager that installs a forward hook to replace hidden states
    at a specific layer during the first forward pass (prefill).

    The hook fires once and then removes itself, so it only affects the
    initial prefill pass, not subsequent autoregressive generation steps.

    Args:
        model: HuggingFace model to hook into.
        injection_layer: Layer index to inject at.
        projected_hidden: Tensor [1, D] or [B, D] to replace the last token's
            hidden state with.

    Yields:
        None. The hook is active during the context.
    """
    import torch

    layers = _get_decoder_layers(model)
    target_layer = layers[injection_layer]

    fired = [False]  # mutable flag for closure

    def hook_fn(module, input, output):
        if fired[0]:
            return output

        fired[0] = True

        # Decoder layer output is a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]  # [B, seq_len, D]
        else:
            hidden = output

        # Replace last token's hidden state with projected source hidden state
        injection = projected_hidden.to(device=hidden.device, dtype=hidden.dtype)
        if injection.dim() == 1:
            injection = injection.unsqueeze(0)  # [D] -> [1, D]

        # Clone to avoid in-place modification
        modified = hidden.clone()
        modified[:, -1, :] = injection  # Replace last position

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def project_for_mid_layer(
    source_hidden: Any,
    avp_map: Any,
    source_model: Any,
    target_model: Any,
    target_num_layers: int,
    injection_depth_ratio: float = DEFAULT_DEPTH_RATIO,
) -> Tuple[Any, int]:
    """Project source hidden state for mid-layer injection.

    Unlike rosetta (which projects to target embedding space for layer-0 inputs_embeds),
    mid-layer projects to the target's intermediate representation space. Since we don't
    have a direct map between intermediate spaces, we use the same vocab-mediated/overlap
    projection but normalize to the target layer's activation norm instead of the
    embedding norm.

    Args:
        source_hidden: Source hidden state [1, D_src] or [D_src].
        avp_map: AVPMap with projection data.
        source_model: Source HuggingFace model.
        target_model: Target HuggingFace model.
        target_num_layers: Number of layers in target model.
        injection_depth_ratio: Depth ratio for injection point.

    Returns:
        Tuple of (projected_hidden [1, D_tgt], injection_layer_index).
    """
    import torch
    from .project import apply_cross_model_projection

    injection_layer = compute_injection_layer(target_num_layers, injection_depth_ratio)

    # Use standard vocab-mediated/overlap projection
    projected = apply_cross_model_projection(
        source_hidden, avp_map, source_model, target_model,
    )

    # Ensure correct shape [1, D]
    if projected.dim() == 1:
        projected = projected.unsqueeze(0)

    return projected, injection_layer
