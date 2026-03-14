"""Forward hooks for trained per-layer cross-model projection.

Installs per-layer forward hooks that additively inject projected source
hidden states (scaled by learned gates) during the first forward pass (prefill).
Hooks fire once and are removed after the context manager exits.
"""

import logging
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_decoder_layers(model: Any):
    """Get the list of decoder layers from a HuggingFace model."""
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return layers

    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        h = getattr(transformer, "h", None)
        if h is not None:
            return h

    raise AttributeError(
        f"Cannot find decoder layers in model {type(model).__name__}. "
        "Expected model.model.layers or model.transformer.h"
    )


@contextmanager
def trained_multi_layer_hook(
    model: Any,
    layer_projections: List[Optional[Tuple[Any, float]]],
):
    """Context manager that installs per-layer forward hooks for trained projection.

    Each hook adds the pre-computed projected hidden state (scaled by gate)
    to the target layer's last-token hidden state. Hooks fire only on the
    first forward pass (prefill), not during autoregressive generation.

    Args:
        model: HuggingFace model to hook into.
        layer_projections: List of length num_layers. Each entry is either:
            - None (gate < threshold, skip this layer)
            - (projected_hidden [1, D_tgt], gate_value float)

    Yields:
        None. Hooks are active during the context.
    """
    import torch

    layers = _get_decoder_layers(model)
    handles = []
    fired_flags = []

    for layer_idx, proj_data in enumerate(layer_projections):
        if proj_data is None:
            continue
        if layer_idx >= len(layers):
            break

        projected, gate = proj_data
        target_layer = layers[layer_idx]
        fired = [False]
        fired_flags.append(fired)

        def make_hook(proj_h, g, f):
            def hook_fn(module, input, output):
                if f[0]:
                    return output

                f[0] = True

                if isinstance(output, tuple):
                    hidden = output[0]  # [B, seq_len, D]
                else:
                    hidden = output

                # Add projected source hidden state (scaled by gate) to last token
                injection = proj_h.to(device=hidden.device, dtype=hidden.dtype)
                if injection.dim() == 2:
                    injection = injection.squeeze(0)  # [1, D] -> [D]

                modified = hidden.clone()
                modified[:, -1, :] = modified[:, -1, :] + g * injection

                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified

            return hook_fn

        hook = make_hook(projected, gate, fired)
        handle = target_layer.register_forward_hook(hook)
        handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
