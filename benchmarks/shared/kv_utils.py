"""KV-cache utility functions for benchmarks."""

from typing import Any

import torch


def get_past_length(past_kv: Any) -> int:
    """Get sequence length from past_key_values (DynamicCache or legacy tuple)."""
    if past_kv is None:
        return 0
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_kv, Cache):
            return past_kv.get_seq_length()
    except ImportError:
        pass
    if isinstance(past_kv, (tuple, list)) and len(past_kv) > 0:
        first = past_kv[0]
        if isinstance(first, (tuple, list)) and len(first) > 0:
            return first[0].shape[-2]
    return 0


def slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    """Slice a KV tensor to keep only the last `tokens_to_keep` positions."""
    if tokens_to_keep <= 0:
        return tensor[..., 0:0, :].contiguous()
    keep = min(tokens_to_keep, tensor.shape[-2])
    start = tensor.shape[-2] - keep
    return tensor[..., start:, :].contiguous()


def truncate_past(past_kv: Any, tokens_to_keep: int) -> Any:
    """Truncate KV-cache to keep only the last `tokens_to_keep` tokens.

    Handles DynamicCache (transformers v5 with .layers API), and legacy
    tuple-of-tuples format.
    """
    if past_kv is None or tokens_to_keep <= 0:
        return None
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            new_cache = DynamicCache()
            for layer in past_kv.layers:
                k = slice_tensor(layer.keys, tokens_to_keep)
                v = slice_tensor(layer.values, tokens_to_keep)
                new_cache.update(k, v, len(new_cache.layers))
            return new_cache
    except ImportError:
        pass
    # Legacy tuple-of-tuples format
    trimmed_layers = []
    for layer in past_kv:
        if isinstance(layer, tuple):
            trimmed_layers.append(tuple(slice_tensor(t, tokens_to_keep) for t in layer))
        elif torch.is_tensor(layer):
            trimmed_layers.append(slice_tensor(layer, tokens_to_keep))
        else:
            trimmed_layers.append(layer)
    return tuple(trimmed_layers)
