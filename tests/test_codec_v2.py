"""Tests for AVP v0.2.0 codec: encode/decode, flags, backward compat."""

import struct

import numpy as np
import pytest

import avp
from avp.codec import _HEADER_FMT, encode, decode, encode_hybrid
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


# --- Hybrid codec ---


class TestHybridCodec:
    """Tests for HYBRID mode: encode_hybrid() and decode() with HybridPayload."""

    def test_encode_hybrid_sets_flag(self):
        """FLAG_HYBRID bit is set in header byte 3."""
        metadata = AVPMetadata(model_id="test")
        data = encode_hybrid(b"\x00", "hello", metadata)
        assert data[3] & FLAG_HYBRID == FLAG_HYBRID

    def test_encode_hybrid_forces_mode(self):
        """metadata.mode is forced to HYBRID even if caller set LATENT."""
        metadata = AVPMetadata(model_id="test", mode=CommunicationMode.LATENT)
        data = encode_hybrid(b"\x00", "hello", metadata)
        msg = decode(data)
        assert msg.metadata.mode == CommunicationMode.HYBRID

    def test_hybrid_roundtrip_text(self):
        """text_fallback survives encode → decode."""
        metadata = AVPMetadata(model_id="test")
        data = encode_hybrid(b"\x00", "summary text here", metadata)
        msg = decode(data)
        assert msg.text_fallback == "summary text here"

    def test_hybrid_roundtrip_latent(self):
        """Latent payload bytes survive encode → decode."""
        latent = b"\xde\xad\xbe\xef" * 256
        metadata = AVPMetadata(model_id="test")
        data = encode_hybrid(latent, "text", metadata)
        msg = decode(data)
        assert msg.payload == latent

    def test_hybrid_roundtrip_kv_cache(self):
        """KV_CACHE flag + payload preserved through HYBRID."""
        kv_data = b"\xaa\xbb\xcc" * 100
        metadata = AVPMetadata(
            model_id="test",
            payload_type=PayloadType.KV_CACHE,
        )
        data = encode_hybrid(kv_data, "kv summary", metadata)
        msg = decode(data)
        assert msg.header.is_hybrid
        assert msg.header.is_kv_cache
        assert msg.payload == kv_data
        assert msg.text_fallback == "kv summary"

    def test_hybrid_empty_text(self):
        """Empty string text_fallback is preserved (not None)."""
        metadata = AVPMetadata(model_id="test")
        data = encode_hybrid(b"\x01", "", metadata)
        msg = decode(data)
        assert msg.text_fallback == ""

    def test_hybrid_unicode_text(self):
        """Non-ASCII text survives UTF-8 roundtrip."""
        text = "Résumé: 数学の問題を解く 🧮"
        metadata = AVPMetadata(model_id="test")
        data = encode_hybrid(b"\x01", text, metadata)
        msg = decode(data)
        assert msg.text_fallback == text

    def test_hybrid_with_compression(self):
        """zstd compression works on HybridPayload."""
        latent = b"\x00" * 4096  # compresses well
        metadata = AVPMetadata(model_id="test")
        data_uncompressed = encode_hybrid(latent, "text", metadata)

        metadata2 = AVPMetadata(model_id="test")
        data_compressed = encode_hybrid(
            latent, "text", metadata2, compression=CompressionLevel.BALANCED
        )
        assert len(data_compressed) < len(data_uncompressed)

        msg = decode(data_compressed)
        assert msg.header.compressed
        assert msg.header.is_hybrid
        assert msg.payload == latent
        assert msg.text_fallback == "text"

    def test_hybrid_confidence_scores(self):
        """Per-chunk confidence values are encoded in the protobuf."""
        from avp import avp_pb2

        metadata = AVPMetadata(model_id="test")
        data = encode_hybrid(
            b"\x01", "text", metadata,
            latent_confidence=0.85,
            text_confidence=0.60,
        )
        # Decode the raw HybridPayload to verify confidence values
        msg = decode(data)
        # Re-parse the wire bytes to check protobuf directly
        raw_msg = decode.__wrapped__ if hasattr(decode, "__wrapped__") else None
        # Just re-encode and check the protobuf layer
        _, _, flags, payload_length, metadata_length = struct.unpack(
            _HEADER_FMT, data[:HEADER_SIZE]
        )
        meta_end = HEADER_SIZE + metadata_length
        raw_payload = data[meta_end:HEADER_SIZE + payload_length]

        hybrid_pb = avp_pb2.HybridPayload()
        hybrid_pb.ParseFromString(raw_payload)
        confidences = {c.chunk_type: c.confidence for c in hybrid_pb.chunks}
        assert confidences[avp_pb2.LATENT_CHUNK] == pytest.approx(0.85, abs=0.01)
        assert confidences[avp_pb2.TEXT_CHUNK] == pytest.approx(0.60, abs=0.01)

    def test_non_hybrid_has_none_fallback(self):
        """Regular LATENT message has text_fallback=None."""
        metadata = AVPMetadata(
            model_id="test",
            mode=CommunicationMode.LATENT,
        )
        data = encode(b"\x00", metadata)
        msg = decode(data)
        assert msg.text_fallback is None
