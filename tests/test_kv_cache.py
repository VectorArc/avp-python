"""Tests for AVP KV-cache serialization (requires torch)."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def test_kv_cache_header_roundtrip():
    from avp.kv_cache import KVCacheHeader

    header = KVCacheHeader(
        num_layers=4,
        num_kv_heads=8,
        head_dim=64,
        seq_len=128,
        dtype="float16",
    )
    data = header.to_bytes()
    restored = KVCacheHeader.from_bytes(data)

    assert restored.num_layers == 4
    assert restored.num_kv_heads == 8
    assert restored.head_dim == 64
    assert restored.seq_len == 128
    assert restored.dtype == "float16"


def test_serialize_deserialize_roundtrip():
    import torch
    from avp.kv_cache import deserialize_kv_cache, serialize_kv_cache

    # Create small random KV-cache: 2 layers, 2 heads, seq_len=4, head_dim=8
    past_key_values = tuple(
        (
            torch.randn(1, 2, 4, 8, dtype=torch.float16),
            torch.randn(1, 2, 4, 8, dtype=torch.float16),
        )
        for _ in range(2)
    )

    data, header = serialize_kv_cache(past_key_values)

    assert header.num_layers == 2
    assert header.num_kv_heads == 2
    assert header.head_dim == 8
    assert header.seq_len == 4
    assert header.dtype == "float16"

    restored, restored_header = deserialize_kv_cache(data)

    assert len(restored) == 2
    for layer_idx in range(2):
        k_orig, v_orig = past_key_values[layer_idx]
        k_rest, v_rest = restored[layer_idx]

        torch.testing.assert_close(k_orig, k_rest, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(v_orig, v_rest, atol=1e-3, rtol=1e-3)


def test_serialize_deserialize_float32():
    import torch
    from avp.kv_cache import deserialize_kv_cache, serialize_kv_cache

    past_key_values = tuple(
        (
            torch.randn(1, 4, 8, 16, dtype=torch.float32),
            torch.randn(1, 4, 8, 16, dtype=torch.float32),
        )
        for _ in range(3)
    )

    data, header = serialize_kv_cache(past_key_values)
    assert header.dtype == "float32"

    restored, _ = deserialize_kv_cache(data)
    assert len(restored) == 3

    for i in range(3):
        torch.testing.assert_close(past_key_values[i][0], restored[i][0])
        torch.testing.assert_close(past_key_values[i][1], restored[i][1])


def test_dynamic_cache_to_legacy():
    import torch
    from avp.kv_cache import dynamic_cache_to_legacy

    # Test with regular tuple format
    past = (
        (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)),
        (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)),
    )

    legacy = dynamic_cache_to_legacy(past)
    assert len(legacy) == 2
    assert len(legacy[0]) == 2


def test_estimate_kv_cache_size():
    from avp.kv_cache import estimate_kv_cache_size

    # 32 layers, 32 heads, 128 head_dim, 100 tokens, float16
    size = estimate_kv_cache_size(
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
        seq_len=100,
        dtype="float16",
    )
    # 2 * 32 * 32 * 100 * 128 * 2 bytes = 52,428,800 = ~50MB
    expected = 2 * 32 * 32 * 100 * 128 * 2
    assert size == expected


def test_empty_kv_cache_raises():
    from avp.kv_cache import serialize_kv_cache

    with pytest.raises(ValueError, match="Empty"):
        serialize_kv_cache(())
