"""Tests for AVP compression."""

import numpy as np
import pytest

import avp
from avp.compression import compress, decompress
from avp.types import AVPMetadata, CompressionLevel, DataType, PayloadType, _STR_TO_DTYPE
from avp.utils import embedding_to_bytes


def _encode_embedding(emb, model_id="", compression=CompressionLevel.NONE):
    """Helper: encode a 1-D numpy embedding with v0.2.0 API."""
    dtype_str = str(emb.dtype)
    dtype_enum = _STR_TO_DTYPE.get(dtype_str, DataType.FLOAT32)
    metadata = AVPMetadata(
        model_id=model_id,
        hidden_dim=emb.shape[0],
        payload_type=PayloadType.EMBEDDING,
        dtype=dtype_enum,
        tensor_shape=emb.shape,
    )
    payload = embedding_to_bytes(emb)
    return avp.encode(payload, metadata, compression)


def test_compress_decompress_roundtrip():
    data = np.random.randn(1024).astype(np.float32).tobytes()
    for level in [CompressionLevel.FAST, CompressionLevel.BALANCED, CompressionLevel.MAX]:
        compressed = compress(data, level)
        restored = decompress(compressed)
        assert restored == data, f"Roundtrip failed at level {level}"


def test_compress_none_is_passthrough():
    data = b"hello world"
    assert compress(data, CompressionLevel.NONE) == data


def test_higher_levels_compress_more():
    # With random-ish float data, higher levels should compress at least as well
    data = np.zeros(4096, dtype=np.float32).tobytes()  # Zeros compress very well
    fast = compress(data, CompressionLevel.FAST)
    balanced = compress(data, CompressionLevel.BALANCED)
    maxx = compress(data, CompressionLevel.MAX)

    assert len(fast) <= len(data)
    assert len(balanced) <= len(fast)
    assert len(maxx) <= len(balanced)


@pytest.mark.parametrize("dim", [384, 1024, 4096])
def test_codec_compression_levels(dim):
    """Full codec roundtrip with each compression level."""
    emb = np.random.randn(dim).astype(np.float32)

    for level in CompressionLevel:
        data = _encode_embedding(emb, model_id="test", compression=level)
        msg = avp.decode(data)
        np.testing.assert_array_equal(emb, msg.embedding)

        if level == CompressionLevel.NONE:
            assert not msg.header.compressed
        else:
            assert msg.header.compressed


def test_compressed_smaller_than_uncompressed():
    """Compressed output should be smaller for sparse/structured data."""
    emb = np.zeros(4096, dtype=np.float32)  # All-zeros compresses extremely well
    raw = _encode_embedding(emb, compression=CompressionLevel.NONE)
    compressed = _encode_embedding(emb, compression=CompressionLevel.FAST)
    assert len(compressed) < len(raw)
