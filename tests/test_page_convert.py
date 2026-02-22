"""Tests for PagedAttention ↔ contiguous tensor conversion (requires torch)."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _make_paged_kv(num_layers, num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32):
    """Create synthetic vLLM-style paged KV blocks.

    Each layer tensor shape: [num_blocks, 2, num_kv_heads, block_size, head_dim]
    """
    return [
        torch.randn(num_blocks, 2, num_kv_heads, block_size, head_dim, dtype=dtype)
        for _ in range(num_layers)
    ]


def _make_block_table(num_blocks):
    """Create a simple block table: logical == physical (identity mapping)."""
    return torch.arange(num_blocks, dtype=torch.long).unsqueeze(0)  # [1, num_blocks]


def test_paged_to_contiguous_basic_shapes():
    from avp.connectors.page_convert import paged_to_contiguous

    num_layers, num_kv_heads, block_size, head_dim = 2, 4, 16, 32
    seq_len = 32  # exactly 2 blocks
    num_blocks = 4  # allocated blocks >= needed blocks

    kv_blocks = _make_paged_kv(num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    block_table = _make_block_table(num_blocks)

    result = paged_to_contiguous(kv_blocks, block_table, num_layers, block_size, seq_len)

    assert len(result) == num_layers
    for k, v in result:
        assert k.shape == (1, num_kv_heads, seq_len, head_dim)
        assert v.shape == (1, num_kv_heads, seq_len, head_dim)


def test_paged_to_contiguous_partial_block():
    """seq_len not a multiple of block_size — last block partially filled."""
    from avp.connectors.page_convert import paged_to_contiguous

    num_layers, num_kv_heads, block_size, head_dim = 2, 4, 16, 32
    seq_len = 25  # 1 full block (16) + 1 partial block (9)
    num_blocks = 4

    kv_blocks = _make_paged_kv(num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    block_table = _make_block_table(num_blocks)

    result = paged_to_contiguous(kv_blocks, block_table, num_layers, block_size, seq_len)

    for k, v in result:
        assert k.shape == (1, num_kv_heads, seq_len, head_dim)
        assert v.shape == (1, num_kv_heads, seq_len, head_dim)


def test_paged_to_contiguous_preserves_values():
    """Verify that gathered contiguous values match the original paged data."""
    from avp.connectors.page_convert import paged_to_contiguous

    num_layers, num_kv_heads, block_size, head_dim = 1, 2, 4, 8
    seq_len = 8  # exactly 2 blocks
    num_blocks = 2

    kv_blocks = _make_paged_kv(num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    block_table = _make_block_table(num_blocks)

    result = paged_to_contiguous(kv_blocks, block_table, num_layers, block_size, seq_len)

    k, v = result[0]
    # First block (phys 0): tokens 0-3
    torch.testing.assert_close(k[0, :, :4, :], kv_blocks[0][0, 0, :, :4, :])
    torch.testing.assert_close(v[0, :, :4, :], kv_blocks[0][0, 1, :, :4, :])
    # Second block (phys 1): tokens 4-7
    torch.testing.assert_close(k[0, :, 4:8, :], kv_blocks[0][1, 0, :, :4, :])
    torch.testing.assert_close(v[0, :, 4:8, :], kv_blocks[0][1, 1, :, :4, :])


def test_paged_to_contiguous_scattered_blocks():
    """Block table with non-sequential physical IDs."""
    from avp.connectors.page_convert import paged_to_contiguous

    num_layers, num_kv_heads, block_size, head_dim = 1, 2, 4, 8
    seq_len = 8
    num_blocks = 4  # more allocated than needed

    kv_blocks = _make_paged_kv(num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    # Scattered: logical block 0 → physical 3, logical block 1 → physical 1
    block_table = torch.tensor([[3, 1, 0, 2]], dtype=torch.long)

    result = paged_to_contiguous(kv_blocks, block_table, num_layers, block_size, seq_len)
    k, v = result[0]

    # First logical block (phys 3): tokens 0-3
    torch.testing.assert_close(k[0, :, :4, :], kv_blocks[0][3, 0, :, :4, :])
    # Second logical block (phys 1): tokens 4-7
    torch.testing.assert_close(k[0, :, 4:8, :], kv_blocks[0][1, 0, :, :4, :])


def test_contiguous_to_paged_basic():
    from avp.connectors.page_convert import contiguous_to_paged

    num_layers, num_kv_heads, block_size, head_dim = 2, 4, 16, 32
    seq_len = 32
    num_blocks = 4

    # Create contiguous KV
    legacy_kv = [
        (torch.randn(1, num_kv_heads, seq_len, head_dim),
         torch.randn(1, num_kv_heads, seq_len, head_dim))
        for _ in range(num_layers)
    ]

    # Pre-allocate paged blocks
    kv_blocks = _make_paged_kv(num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    block_table = _make_block_table(num_blocks)

    contiguous_to_paged(legacy_kv, kv_blocks, block_table, block_size)

    # Verify block 0, layer 0 keys match first block_size tokens
    k_orig = legacy_kv[0][0].squeeze(0)  # [num_kv_heads, seq_len, head_dim]
    torch.testing.assert_close(kv_blocks[0][0, 0, :, :block_size, :], k_orig[:, :block_size, :])


def test_roundtrip_paged_contiguous_paged():
    """paged → contiguous → paged round-trip preserves data."""
    from avp.connectors.page_convert import contiguous_to_paged, paged_to_contiguous

    num_layers, num_kv_heads, block_size, head_dim = 2, 4, 8, 16
    seq_len = 20  # partial last block (20 / 8 = 2 full + 4 leftover)
    num_blocks = 4

    original_kv = _make_paged_kv(num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    block_table = _make_block_table(num_blocks)

    # paged → contiguous
    contiguous = paged_to_contiguous(original_kv, block_table, num_layers, block_size, seq_len)

    # contiguous → paged (into fresh blocks)
    restored_kv = [
        torch.zeros_like(original_kv[i]) for i in range(num_layers)
    ]
    contiguous_to_paged(contiguous, restored_kv, block_table, block_size)

    # Compare used blocks only
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    for layer_idx in range(num_layers):
        for block_i in range(num_blocks_needed):
            phys_id = block_table[0, block_i].item()
            tok_start = block_i * block_size
            tok_end = min(tok_start + block_size, seq_len)
            actual_tokens = tok_end - tok_start

            torch.testing.assert_close(
                restored_kv[layer_idx][phys_id, 0, :, :actual_tokens, :],
                original_kv[layer_idx][phys_id, 0, :, :actual_tokens, :],
            )
            torch.testing.assert_close(
                restored_kv[layer_idx][phys_id, 1, :, :actual_tokens, :],
                original_kv[layer_idx][phys_id, 1, :, :actual_tokens, :],
            )


def test_roundtrip_with_serialization():
    """paged → contiguous → serialize → deserialize → paged round-trip."""
    from avp.connectors.page_convert import contiguous_to_paged, paged_to_contiguous
    from avp.kv_cache import deserialize_kv_cache, serialize_kv_cache

    num_layers, num_kv_heads, block_size, head_dim = 2, 2, 4, 8
    seq_len = 8
    num_blocks = 2

    original_kv = _make_paged_kv(
        num_layers, num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16
    )
    block_table = _make_block_table(num_blocks)

    # paged → contiguous
    contiguous = paged_to_contiguous(original_kv, block_table, num_layers, block_size, seq_len)

    # contiguous → serialize → deserialize
    legacy_tuple = tuple(contiguous)
    data, header = serialize_kv_cache(legacy_tuple)
    restored_legacy, _ = deserialize_kv_cache(data)

    # deserialize → paged
    restored_paged = _make_paged_kv(
        num_layers, num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16
    )
    contiguous_to_paged(list(restored_legacy), restored_paged, block_table, block_size)

    # Compare
    for layer_idx in range(num_layers):
        for block_i in range(num_blocks):
            torch.testing.assert_close(
                restored_paged[layer_idx][block_i],
                original_kv[layer_idx][block_i],
                atol=1e-3, rtol=1e-3,
            )


def test_single_layer():
    from avp.connectors.page_convert import paged_to_contiguous

    kv_blocks = _make_paged_kv(1, 2, 2, 4, 8)
    block_table = _make_block_table(2)
    result = paged_to_contiguous(kv_blocks, block_table, 1, 4, 8)
    assert len(result) == 1
    assert result[0][0].shape == (1, 2, 8, 8)


def test_many_layers():
    from avp.connectors.page_convert import paged_to_contiguous

    kv_blocks = _make_paged_kv(32, 4, 8, 16, 64)
    block_table = _make_block_table(4)
    result = paged_to_contiguous(kv_blocks, block_table, 32, 16, 48)
    assert len(result) == 32
    for k, v in result:
        assert k.shape == (1, 8, 48, 64)


def test_float16_dtype():
    from avp.connectors.page_convert import paged_to_contiguous

    kv_blocks = _make_paged_kv(1, 2, 2, 4, 8, dtype=torch.float16)
    block_table = _make_block_table(2)
    result = paged_to_contiguous(kv_blocks, block_table, 1, 4, 8)
    assert result[0][0].dtype == torch.float16


def test_bfloat16_dtype():
    from avp.connectors.page_convert import paged_to_contiguous

    kv_blocks = _make_paged_kv(1, 2, 2, 4, 8, dtype=torch.bfloat16)
    block_table = _make_block_table(2)
    result = paged_to_contiguous(kv_blocks, block_table, 1, 4, 8)
    assert result[0][0].dtype == torch.bfloat16


def test_1d_block_table():
    """block_table can be 1D (single sequence, no batch dim)."""
    from avp.connectors.page_convert import paged_to_contiguous

    kv_blocks = _make_paged_kv(1, 2, 2, 4, 8)
    block_table = torch.arange(2, dtype=torch.long)  # 1D
    result = paged_to_contiguous(kv_blocks, block_table, 1, 4, 8)
    assert result[0][0].shape == (1, 2, 8, 8)


def test_invalid_params():
    from avp.connectors.page_convert import paged_to_contiguous

    kv_blocks = _make_paged_kv(1, 2, 2, 4, 8)
    block_table = _make_block_table(2)

    with pytest.raises(ValueError, match="num_layers"):
        paged_to_contiguous(kv_blocks, block_table, 0, 4, 8)
    with pytest.raises(ValueError, match="block_size"):
        paged_to_contiguous(kv_blocks, block_table, 1, 0, 8)
    with pytest.raises(ValueError, match="seq_len"):
        paged_to_contiguous(kv_blocks, block_table, 1, 4, 0)


def test_contiguous_to_paged_empty_raises():
    from avp.connectors.page_convert import contiguous_to_paged

    with pytest.raises(ValueError, match="Empty"):
        contiguous_to_paged([], [], torch.tensor([0]), 4)
