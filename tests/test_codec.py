"""Tests for AVP encode/decode roundtrips."""

import struct

import numpy as np
import pytest

import avp
from avp.codec import _HEADER_FMT
from avp.types import (
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    AVPMetadata,
    DataType,
    PayloadType,
    _STR_TO_DTYPE,
)
from avp.utils import embedding_to_bytes


def _encode_embedding(emb, model_id="", compression=avp.CompressionLevel.NONE,
                      agent_id="", extra=None):
    """Helper: encode a 1-D numpy embedding."""
    dtype_str = str(emb.dtype)
    dtype_enum = _STR_TO_DTYPE.get(dtype_str, DataType.FLOAT32)
    metadata = AVPMetadata(
        model_id=model_id,
        hidden_dim=emb.shape[0],
        payload_type=PayloadType.EMBEDDING,
        dtype=dtype_enum,
        tensor_shape=emb.shape,
        source_agent_id=agent_id,
        extra=extra or {},
    )
    payload = embedding_to_bytes(emb)
    return avp.encode(payload, metadata, compression)


# --- Roundtrip tests across dimensions and dtypes ---


@pytest.mark.parametrize("dim", [384, 768, 1024, 4096])
def test_roundtrip_float32(dim):
    emb = np.random.randn(dim).astype(np.float32)
    data = _encode_embedding(emb, model_id=f"model-{dim}")
    msg = avp.decode(data)

    assert msg.metadata.model_id == f"model-{dim}"
    assert msg.metadata.embedding_dim == dim
    assert msg.metadata.data_type == "float32"
    assert msg.embedding.shape == (dim,)
    np.testing.assert_array_equal(emb, msg.embedding)


@pytest.mark.parametrize("dim", [384, 768, 1024, 4096])
def test_roundtrip_float16(dim):
    emb = np.random.randn(dim).astype(np.float16)
    data = _encode_embedding(emb, model_id=f"model-fp16-{dim}")
    msg = avp.decode(data)

    assert msg.metadata.data_type == "float16"
    assert msg.embedding.dtype == np.float16
    np.testing.assert_array_equal(emb, msg.embedding)


def test_roundtrip_with_all_metadata():
    emb = np.random.randn(384).astype(np.float32)
    data = _encode_embedding(
        emb,
        model_id="all-MiniLM-L6-v2",
        agent_id="agent-alice",
        extra={"session": "abc", "priority": "high", "task_id": "task-123"},
    )
    msg = avp.decode(data)

    assert msg.metadata.model_id == "all-MiniLM-L6-v2"
    assert msg.metadata.agent_id == "agent-alice"
    assert msg.metadata.task_id == "task-123"
    assert msg.metadata.extra["session"] == "abc"
    assert msg.metadata.extra["priority"] == "high"
    np.testing.assert_array_equal(emb, msg.embedding)


# --- Header validation ---


def test_header_magic_bytes():
    emb = np.zeros(10, dtype=np.float32)
    data = _encode_embedding(emb)
    assert data[:2] == MAGIC


def test_header_version():
    emb = np.zeros(10, dtype=np.float32)
    data = _encode_embedding(emb)
    assert data[2] == PROTOCOL_VERSION


def test_header_uncompressed_flag():
    emb = np.zeros(10, dtype=np.float32)
    data = _encode_embedding(emb)
    assert data[3] == 0x00  # No compression flag


def test_header_compressed_flag():
    emb = np.zeros(10, dtype=np.float32)
    data = _encode_embedding(emb, compression=avp.CompressionLevel.FAST)
    assert data[3] & 0x01 == 0x01


def test_header_payload_length():
    emb = np.zeros(100, dtype=np.float32)
    data = _encode_embedding(emb)
    _, _, _, payload_len, meta_len = struct.unpack(_HEADER_FMT, data[:HEADER_SIZE])
    assert payload_len == len(data) - HEADER_SIZE
    assert meta_len < payload_len


# --- Error handling ---


def test_invalid_magic():
    bad = b"\x00\x00" + b"\x01\x00" + b"\x00" * 8
    with pytest.raises(avp.InvalidMagicError):
        avp.decode(bad)


def test_unsupported_version():
    bad = MAGIC + b"\xff\x00" + b"\x00" * 8
    with pytest.raises(avp.UnsupportedVersionError):
        avp.decode(bad)


def test_truncated_message():
    with pytest.raises(avp.DecodeError):
        avp.decode(b"\x41\x56\x01")


def test_message_too_short():
    emb = np.zeros(10, dtype=np.float32)
    data = _encode_embedding(emb)
    with pytest.raises(avp.DecodeError):
        avp.decode(data[:HEADER_SIZE + 1])


# --- Size comparison ---


def test_binary_smaller_than_json():
    """AVP binary should be significantly smaller than JSON."""
    from avp.utils import embedding_to_json

    emb = np.random.randn(4096).astype(np.float32)
    avp_data = _encode_embedding(emb, model_id="test")
    json_data = embedding_to_json(emb, {"model_id": "test"})

    # AVP should be at least 5x smaller than JSON
    assert len(avp_data) < len(json_data) / 5
