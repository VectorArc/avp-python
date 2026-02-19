"""Tests for AVP v0.2.0 types, enums, and dataclasses."""

import numpy as np
import pytest

from avp.types import (
    FLAG_COMPRESSED,
    FLAG_HAS_MAP,
    FLAG_HYBRID,
    FLAG_KV_CACHE,
    PROTOCOL_VERSION,
    AVP_VERSION_STRING,
    AVPHeader,
    AVPMessage,
    AVPMetadata,
    CommunicationMode,
    DataType,
    ModelIdentity,
    PayloadType,
    SessionInfo,
    _DTYPE_TO_STR,
    _STR_TO_DTYPE,
)


# --- Protocol version constants ---


def test_protocol_version():
    assert PROTOCOL_VERSION == 0x01


def test_avp_version_string():
    assert AVP_VERSION_STRING == "0.2.0"


# --- Enums ---


def test_payload_type_values():
    assert PayloadType.HIDDEN_STATE == 0
    assert PayloadType.KV_CACHE == 1
    assert PayloadType.EMBEDDING == 2


def test_communication_mode_values():
    assert CommunicationMode.LATENT == 0
    assert CommunicationMode.HYBRID == 1
    assert CommunicationMode.JSON == 2


def test_data_type_values():
    assert DataType.FLOAT32 == 0
    assert DataType.FLOAT16 == 1
    assert DataType.BFLOAT16 == 2
    assert DataType.INT8 == 3


def test_dtype_str_mapping():
    assert _DTYPE_TO_STR[DataType.FLOAT32] == "float32"
    assert _DTYPE_TO_STR[DataType.FLOAT16] == "float16"
    assert _DTYPE_TO_STR[DataType.BFLOAT16] == "bfloat16"
    assert _DTYPE_TO_STR[DataType.INT8] == "int8"
    # Roundtrip
    for dtype_enum, dtype_str in _DTYPE_TO_STR.items():
        assert _STR_TO_DTYPE[dtype_str] == dtype_enum


# --- Flag constants ---


def test_flag_constants():
    assert FLAG_COMPRESSED == 0x01
    assert FLAG_HYBRID == 0x02
    assert FLAG_HAS_MAP == 0x04
    assert FLAG_KV_CACHE == 0x08


# --- AVPHeader ---


def test_header_flag_properties():
    h = AVPHeader(flags=0)
    assert not h.compressed
    assert not h.is_hybrid
    assert not h.has_map
    assert not h.is_kv_cache

    h.compressed = True
    assert h.compressed
    assert h.flags & FLAG_COMPRESSED

    h = AVPHeader(flags=FLAG_HYBRID | FLAG_HAS_MAP | FLAG_KV_CACHE)
    assert h.is_hybrid
    assert h.has_map
    assert h.is_kv_cache
    assert not h.compressed


def test_header_compressed_setter():
    h = AVPHeader(flags=0)
    h.compressed = True
    assert h.flags == FLAG_COMPRESSED
    h.compressed = False
    assert h.flags == 0


# --- ModelIdentity ---


def test_model_identity_to_from_dict():
    identity = ModelIdentity(
        model_family="llama",
        model_id="meta-llama/Llama-2-7b",
        model_hash="abc123",
        hidden_dim=4096,
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
    )
    d = identity.to_dict()
    restored = ModelIdentity.from_dict(d)
    assert restored.model_family == "llama"
    assert restored.model_id == "meta-llama/Llama-2-7b"
    assert restored.model_hash == "abc123"
    assert restored.hidden_dim == 4096
    assert restored.num_layers == 32
    assert restored.num_kv_heads == 32
    assert restored.head_dim == 128


def test_model_identity_from_empty_dict():
    identity = ModelIdentity.from_dict({})
    assert identity.model_family == ""
    assert identity.hidden_dim == 0


# --- SessionInfo ---


def test_session_info_defaults():
    info = SessionInfo()
    assert info.session_id == ""
    assert info.mode == CommunicationMode.JSON


# --- AVPMetadata ---


def test_metadata_backward_compat():
    meta = AVPMetadata(
        model_id="test",
        hidden_dim=384,
        dtype=DataType.FLOAT16,
        source_agent_id="alice",
        extra={"task_id": "t123"},
    )
    assert meta.embedding_dim == 384
    assert meta.data_type == "float16"
    assert meta.agent_id == "alice"
    assert meta.task_id == "t123"


def test_metadata_agent_id_none_when_empty():
    meta = AVPMetadata()
    assert meta.agent_id is None
    assert meta.task_id is None


# --- AVPMessage ---


def test_message_embedding_property():
    payload = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    meta = AVPMetadata(hidden_dim=3, dtype=DataType.FLOAT32)
    msg = AVPMessage(header=AVPHeader(), metadata=meta, payload=payload)
    emb = msg.embedding
    np.testing.assert_array_equal(emb, [1.0, 2.0, 3.0])


def test_message_empty_payload_embedding():
    msg = AVPMessage(header=AVPHeader(), metadata=AVPMetadata(), payload=b"")
    emb = msg.embedding
    assert emb.shape == (0,)
