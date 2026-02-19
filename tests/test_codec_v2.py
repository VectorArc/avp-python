"""Tests for AVP v0.2.0 codec: encode/decode, flags, backward compat."""

import struct

import numpy as np
import pytest

import avp
from avp.codec import _HEADER_FMT, encode, decode
from avp.types import (
    FLAG_COMPRESSED,
    FLAG_HAS_MAP,
    FLAG_HYBRID,
    FLAG_KV_CACHE,
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    AVPMetadata,
    CommunicationMode,
    CompressionLevel,
    DataType,
    PayloadType,
)
from avp.utils import embedding_to_bytes


# --- v0.2.0 encode/decode roundtrip ---


def test_encode_decode_hidden_state():
    """Roundtrip a hidden state payload."""
    hidden = np.random.randn(4096).astype(np.float32)
    metadata = AVPMetadata(
        session_id="sess-1",
        source_agent_id="agent-a",
        target_agent_id="agent-b",
        model_id="llama-7b",
        hidden_dim=4096,
        num_layers=32,
        payload_type=PayloadType.HIDDEN_STATE,
        dtype=DataType.FLOAT32,
        tensor_shape=(4096,),
        mode=CommunicationMode.LATENT,
    )
    payload = embedding_to_bytes(hidden)
    data = encode(payload, metadata)
    msg = decode(data)

    assert msg.metadata.session_id == "sess-1"
    assert msg.metadata.source_agent_id == "agent-a"
    assert msg.metadata.target_agent_id == "agent-b"
    assert msg.metadata.model_id == "llama-7b"
    assert msg.metadata.hidden_dim == 4096
    assert msg.metadata.num_layers == 32
    assert msg.metadata.payload_type == PayloadType.HIDDEN_STATE
    assert msg.metadata.dtype == DataType.FLOAT32
    assert msg.metadata.tensor_shape == (4096,)
    assert msg.metadata.mode == CommunicationMode.LATENT
    assert msg.header.version == PROTOCOL_VERSION
    np.testing.assert_array_equal(hidden, msg.embedding)


def test_encode_decode_float16():
    hidden = np.random.randn(768).astype(np.float16)
    metadata = AVPMetadata(
        model_id="qwen-0.5b",
        hidden_dim=768,
        payload_type=PayloadType.HIDDEN_STATE,
        dtype=DataType.FLOAT16,
        tensor_shape=(768,),
    )
    payload = embedding_to_bytes(hidden)
    data = encode(payload, metadata)
    msg = decode(data)

    assert msg.metadata.dtype == DataType.FLOAT16
    assert msg.metadata.data_type == "float16"
    np.testing.assert_array_equal(hidden, msg.embedding)


def test_encode_decode_with_compression():
    hidden = np.zeros(2048, dtype=np.float32)  # Compresses well
    metadata = AVPMetadata(
        model_id="test",
        hidden_dim=2048,
        payload_type=PayloadType.HIDDEN_STATE,
        dtype=DataType.FLOAT32,
        tensor_shape=(2048,),
    )
    payload = embedding_to_bytes(hidden)

    data_uncompressed = encode(payload, metadata, CompressionLevel.NONE)
    data_compressed = encode(payload, metadata, CompressionLevel.BALANCED)

    assert len(data_compressed) < len(data_uncompressed)

    msg = decode(data_compressed)
    assert msg.header.compressed
    assert msg.metadata.compression == "zstd"
    np.testing.assert_array_equal(hidden, msg.embedding)


def test_encode_decode_kv_cache_payload():
    """KV-cache payload type sets the FLAG_KV_CACHE bit."""
    kv_data = b"\x00" * 1024
    metadata = AVPMetadata(
        model_id="test",
        payload_type=PayloadType.KV_CACHE,
        dtype=DataType.FLOAT16,
    )
    data = encode(kv_data, metadata)
    msg = decode(data)

    assert msg.header.is_kv_cache
    assert msg.metadata.payload_type == PayloadType.KV_CACHE
    assert msg.payload == kv_data


def test_encode_decode_extra_fields():
    payload = b"\x01\x02\x03"
    metadata = AVPMetadata(
        model_id="test",
        confidence_score=0.95,
        avp_map_id="map-xyz",
        extra={"custom_key": "custom_value"},
    )
    data = encode(payload, metadata)
    msg = decode(data)

    assert msg.metadata.confidence_score == pytest.approx(0.95, abs=0.01)
    assert msg.metadata.avp_map_id == "map-xyz"
    assert msg.metadata.extra["custom_key"] == "custom_value"


# --- Flag bits ---


def test_flag_bits_hybrid():
    payload = b"\x00"
    metadata = AVPMetadata(
        mode=CommunicationMode.HYBRID,
    )
    data = encode(payload, metadata)
    assert data[3] & FLAG_HYBRID == FLAG_HYBRID

    msg = decode(data)
    assert msg.header.is_hybrid
    assert msg.metadata.mode == CommunicationMode.HYBRID


def test_flag_bits_has_map():
    payload = b"\x00"
    metadata = AVPMetadata(
        avp_map_id="some-map",
    )
    data = encode(payload, metadata)
    assert data[3] & FLAG_HAS_MAP == FLAG_HAS_MAP

    msg = decode(data)
    assert msg.header.has_map


def test_flag_bits_kv_cache():
    payload = b"\x00"
    metadata = AVPMetadata(
        payload_type=PayloadType.KV_CACHE,
    )
    data = encode(payload, metadata)
    assert data[3] & FLAG_KV_CACHE == FLAG_KV_CACHE

    msg = decode(data)
    assert msg.header.is_kv_cache


def test_flag_bits_combined():
    payload = b"\x00"
    metadata = AVPMetadata(
        payload_type=PayloadType.KV_CACHE,
        mode=CommunicationMode.HYBRID,
        avp_map_id="map-1",
    )
    data = encode(payload, metadata, CompressionLevel.FAST)

    flags = data[3]
    assert flags & FLAG_COMPRESSED
    assert flags & FLAG_HYBRID
    assert flags & FLAG_HAS_MAP
    assert flags & FLAG_KV_CACHE


# --- Convenience encoders ---


def test_encode_hidden_state():
    hidden = np.random.randn(512).astype(np.float32)
    metadata = AVPMetadata(model_id="test", hidden_dim=512)
    data = avp.encode_hidden_state(hidden, metadata)
    msg = avp.decode(data)

    assert msg.metadata.payload_type == PayloadType.HIDDEN_STATE
    assert msg.metadata.tensor_shape == (512,)
    np.testing.assert_array_equal(hidden, msg.embedding)


def test_encode_kv_cache():
    kv_data = b"\xaa\xbb\xcc" * 100
    metadata = AVPMetadata(model_id="test")
    data = avp.encode_kv_cache(kv_data, metadata)
    msg = avp.decode(data)

    assert msg.metadata.payload_type == PayloadType.KV_CACHE
    assert msg.payload == kv_data
