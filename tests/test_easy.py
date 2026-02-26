"""Tests for the avp.pack() / avp.unpack() easy API.

Layer 0/1 tests — runs with zero optional deps (no torch/transformers).
"""

import json

import pytest

from avp.easy import PackedMessage, clear_cache, pack, unpack
from avp.types import AVP_VERSION_HEADER


# ---------------------------------------------------------------------------
# PackedMessage
# ---------------------------------------------------------------------------


class TestPackedMessage:
    def test_str(self):
        msg = PackedMessage(content="hello")
        assert str(msg) == "hello"

    def test_bytes_json(self):
        msg = PackedMessage(content="hello")
        raw = bytes(msg)
        d = json.loads(raw)
        assert d["avp"] == AVP_VERSION_HEADER
        assert d["content"] == "hello"
        assert "identity" not in d

    def test_bytes_with_identity(self):
        msg = PackedMessage(
            content="hello",
            identity={"model_hash": "abc", "hidden_dim": 4096},
        )
        raw = bytes(msg)
        d = json.loads(raw)
        assert d["identity"]["model_hash"] == "abc"
        assert d["identity"]["hidden_dim"] == 4096

    def test_to_bytes_equals_bytes(self):
        msg = PackedMessage(content="test")
        assert msg.to_bytes() == bytes(msg)


# ---------------------------------------------------------------------------
# pack() — Layer 0 (no model)
# ---------------------------------------------------------------------------


class TestPackLayer0:
    def test_basic_pack(self):
        msg = pack("Hello from agent A")
        assert msg.content == "Hello from agent A"
        assert msg.identity is None
        assert msg.context is None
        assert msg.model is None

    def test_pack_returns_packed_message(self):
        msg = pack("test")
        assert isinstance(msg, PackedMessage)

    def test_pack_str(self):
        msg = pack("content")
        assert str(msg) == "content"

    def test_pack_wire_format(self):
        msg = pack("wire test")
        raw = msg.to_bytes()
        d = json.loads(raw)
        assert d == {"avp": AVP_VERSION_HEADER, "content": "wire test"}


# ---------------------------------------------------------------------------
# unpack() — Layer 0 (extract text)
# ---------------------------------------------------------------------------


class TestUnpackLayer0:
    def test_unpack_bytes_json(self):
        raw = json.dumps({"avp": "0.2", "content": "hello"}).encode()
        assert unpack(raw) == "hello"

    def test_unpack_str_json(self):
        raw = json.dumps({"avp": "0.2", "content": "from json"})
        assert unpack(raw) == "from json"

    def test_unpack_packed_message(self):
        msg = PackedMessage(content="direct")
        assert unpack(msg) == "direct"

    def test_unpack_plain_text(self):
        assert unpack("just text") == "just text"

    def test_unpack_plain_bytes(self):
        assert unpack(b"plain bytes") == "plain bytes"

    def test_unpack_with_identity(self):
        raw = json.dumps({
            "avp": "0.2",
            "content": "with id",
            "identity": {"model_hash": "xyz"},
        }).encode()
        assert unpack(raw) == "with id"

    def test_unpack_invalid_type(self):
        with pytest.raises(TypeError, match="expects bytes, str, or PackedMessage"):
            unpack(12345)


# ---------------------------------------------------------------------------
# Roundtrip: pack → to_bytes → unpack
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_json_roundtrip(self):
        msg = pack("roundtrip test")
        raw = msg.to_bytes()
        result = unpack(raw)
        assert result == "roundtrip test"

    def test_json_roundtrip_with_identity(self):
        msg = PackedMessage(
            content="with identity",
            identity={"model_hash": "abc", "hidden_dim": 1024},
        )
        raw = msg.to_bytes()
        result = unpack(raw)
        assert result == "with identity"

    def test_str_roundtrip(self):
        msg = pack("text roundtrip")
        text = str(msg)
        result = unpack(text)
        assert result == "text roundtrip"


# ---------------------------------------------------------------------------
# Backward compatibility with JSONMessage
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_unpack_legacy_json_message(self):
        """unpack() should handle legacy JSONMessage format (avp_version key)."""
        legacy = json.dumps({
            "avp_version": "0.2.0",
            "session_id": "sess-1",
            "source_agent_id": "alice",
            "target_agent_id": "bob",
            "content": "legacy message",
        })
        result = unpack(legacy)
        assert result == "legacy message"

    def test_unpack_legacy_json_message_bytes(self):
        legacy = json.dumps({
            "avp_version": "0.2.0",
            "content": "legacy bytes",
        }).encode()
        result = unpack(legacy)
        assert result == "legacy bytes"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_content(self):
        msg = pack("")
        assert msg.content == ""
        assert unpack(msg.to_bytes()) == ""

    def test_unicode_content(self):
        text = "Hello 🌍 こんにちは"
        msg = pack(text)
        assert unpack(msg.to_bytes()) == text

    def test_json_without_avp_key(self):
        """Non-AVP JSON is returned as-is (raw text)."""
        raw = json.dumps({"foo": "bar"})
        result = unpack(raw)
        assert result == raw

    def test_unpack_malformed_json(self):
        """Malformed JSON-like string is returned as plain text."""
        result = unpack("{not valid json")
        assert result == "{not valid json"

    def test_clear_cache(self):
        # Should not raise even when caches are empty
        clear_cache()

    def test_whitespace_before_json(self):
        raw = '  \n {"avp": "0.2", "content": "whitespace"}'
        assert unpack(raw) == "whitespace"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_pack_rejects_non_string_content(self):
        with pytest.raises(TypeError, match="content must be str"):
            pack(123)

    def test_pack_rejects_none_content(self):
        with pytest.raises(TypeError, match="content must be str"):
            pack(None)

    def test_pack_rejects_list_content(self):
        with pytest.raises(TypeError, match="content must be str"):
            pack(["a", "b"])

    def test_pack_think_steps_requires_model(self):
        with pytest.raises(ValueError, match="think_steps requires model="):
            pack("hello", think_steps=20)

    def test_unpack_bytearray(self):
        raw = bytearray(json.dumps({"avp": "0.2", "content": "from buffer"}).encode())
        assert unpack(raw) == "from buffer"

    def test_unpack_memoryview(self):
        raw = memoryview(json.dumps({"avp": "0.2", "content": "from mv"}).encode())
        assert unpack(raw) == "from mv"

    def test_from_wire_json(self):
        raw = json.dumps({
            "avp": "0.2",
            "content": "hello",
            "identity": {"model_hash": "abc"},
        }).encode()
        msg = PackedMessage.from_wire(raw)
        assert msg.content == "hello"
        assert msg.identity == {"model_hash": "abc"}

    def test_from_wire_plain_text(self):
        msg = PackedMessage.from_wire("just text")
        assert msg.content == "just text"
        assert msg.identity is None
