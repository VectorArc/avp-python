"""PagedAttention ↔ contiguous tensor conversion for vLLM integration.

vLLM uses PagedAttention where KV-cache is stored in scattered GPU memory blocks.
AVP uses contiguous tensors (matching HuggingFace's format). This module converts
between the two representations.

Pure torch operations — no vLLM dependency. Testable on any platform.
"""

from typing import Any, List, Tuple


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "torch is required for page conversion. Install with: pip install avp[latent]"
        )


def paged_to_contiguous(
    kv_cache_blocks: Any,
    block_table: Any,
    num_layers: int,
    block_size: int,
    seq_len: int,
) -> List[Tuple[Any, Any]]:
    """Convert vLLM paged KV-cache to contiguous tensors.

    vLLM stores KV-cache in fixed-size blocks scattered across GPU memory.
    Each block holds `block_size` tokens. The block_table maps logical block
    indices to physical block IDs.

    Args:
        kv_cache_blocks: List of per-layer KV tensors.
            Each tensor shape: [num_blocks, 2, num_kv_heads, block_size, head_dim]
            Index 0 = keys, index 1 = values.
        block_table: Tensor mapping logical to physical block IDs.
            Shape: [num_seqs, max_blocks_per_seq]. We use the first sequence (index 0).
        num_layers: Number of transformer layers.
        block_size: Tokens per block (typically 16).
        seq_len: Actual sequence length (may not fill the last block).

    Returns:
        List of (K, V) tuples per layer.
        Each K/V shape: [1, num_kv_heads, seq_len, head_dim]
        Compatible with kv_cache.serialize_kv_cache().
    """
    torch = _require_torch()

    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")

    # Get block IDs for the first sequence
    if block_table.dim() == 2:
        seq_block_ids = block_table[0]
    else:
        seq_block_ids = block_table

    num_blocks_needed = (seq_len + block_size - 1) // block_size

    layers = []
    for layer_idx in range(num_layers):
        layer_kv = kv_cache_blocks[layer_idx]
        # layer_kv shape: [num_blocks, 2, num_kv_heads, block_size, head_dim]
        num_kv_heads = layer_kv.shape[2]
        head_dim = layer_kv.shape[4]

        # Gather blocks in logical order
        k_blocks = []
        v_blocks = []
        for block_i in range(num_blocks_needed):
            phys_id = seq_block_ids[block_i].item()
            # Keys: index 0, Values: index 1
            k_blocks.append(layer_kv[phys_id, 0])  # [num_kv_heads, block_size, head_dim]
            v_blocks.append(layer_kv[phys_id, 1])

        # Concatenate along token dimension and trim to actual seq_len
        k_cat = torch.cat(k_blocks, dim=1)[:, :seq_len, :]  # [num_kv_heads, seq_len, head_dim]
        v_cat = torch.cat(v_blocks, dim=1)[:, :seq_len, :]

        # Add batch dimension
        layers.append((k_cat.unsqueeze(0), v_cat.unsqueeze(0)))

    return layers


def contiguous_to_paged(
    legacy_kv: List[Tuple[Any, Any]],
    kv_cache_blocks: Any,
    block_table: Any,
    block_size: int,
) -> None:
    """Write contiguous KV-cache tensors into vLLM paged blocks.

    Scatters contiguous K/V tensors into the pre-allocated paged KV-cache
    buffers using the block_table mapping.

    Args:
        legacy_kv: List of (K, V) tuples from deserialize_kv_cache().
            Each K/V shape: [1, num_kv_heads, seq_len, head_dim]
        kv_cache_blocks: List of per-layer KV tensors (pre-allocated by vLLM).
            Each tensor shape: [num_blocks, 2, num_kv_heads, block_size, head_dim]
            Modified in-place.
        block_table: Tensor mapping logical to physical block IDs.
            Shape: [num_seqs, max_blocks_per_seq]. We use the first sequence (index 0).
        block_size: Tokens per block.
    """
    torch = _require_torch()

    if not legacy_kv:
        raise ValueError("Empty KV-cache")

    # Get block IDs for the first sequence
    if block_table.dim() == 2:
        seq_block_ids = block_table[0]
    else:
        seq_block_ids = block_table

    seq_len = legacy_kv[0][0].shape[2]
    num_blocks_needed = (seq_len + block_size - 1) // block_size

    for layer_idx, (k, v) in enumerate(legacy_kv):
        # k, v shape: [1, num_kv_heads, seq_len, head_dim]
        k_sq = k.squeeze(0)  # [num_kv_heads, seq_len, head_dim]
        v_sq = v.squeeze(0)

        layer_kv = kv_cache_blocks[layer_idx]

        for block_i in range(num_blocks_needed):
            phys_id = seq_block_ids[block_i].item()
            tok_start = block_i * block_size
            tok_end = min(tok_start + block_size, seq_len)
            actual_tokens = tok_end - tok_start

            # Write into the paged block
            layer_kv[phys_id, 0, :, :actual_tokens, :] = k_sq[:, tok_start:tok_end, :]
            layer_kv[phys_id, 1, :, :actual_tokens, :] = v_sq[:, tok_start:tok_end, :]
