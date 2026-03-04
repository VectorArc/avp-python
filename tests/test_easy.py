"""Tests for the avp easy API: think(), generate().

Tests for deprecated pack()/unpack() are in test_deprecated.py.
"""

import json

from avp.easy import PackedMessage, clear_cache
from avp.types import AVP_VERSION_HEADER


# ---------------------------------------------------------------------------
# PackedMessage (still importable, not deprecated on import)
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
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clear_cache(self):
        # Should not raise even when caches are empty
        clear_cache()
