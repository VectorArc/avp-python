"""Tests that deprecated pack()/unpack() emit DeprecationWarning."""

import json
import warnings

from avp.easy import PackedMessage


class TestPackDeprecated:
    def test_pack_emits_deprecation_warning(self):
        from avp.easy import pack

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pack("hello")
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
            assert any("avp.pack()" in str(x.message) for x in w)

    def test_pack_still_works(self):
        from avp.easy import pack

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            msg = pack("hello")
        assert isinstance(msg, PackedMessage)
        assert msg.content == "hello"

    def test_pack_wire_format_still_works(self):
        from avp.easy import pack

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            msg = pack("wire test")
        raw = msg.to_bytes()
        d = json.loads(raw)
        assert d["content"] == "wire test"


class TestUnpackDeprecated:
    def test_unpack_emits_deprecation_warning(self):
        from avp.easy import unpack

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            unpack("hello")
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
            assert any("avp.unpack()" in str(x.message) for x in w)

    def test_unpack_still_extracts_text(self):
        from avp.easy import unpack

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = unpack("just text")
        assert result == "just text"

    def test_unpack_json_still_works(self):
        from avp.easy import unpack

        raw = json.dumps({"avp": "0.2", "content": "hello"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = unpack(raw)
        assert result == "hello"

    def test_unpack_packed_message_still_works(self):
        from avp.easy import unpack

        msg = PackedMessage(content="direct")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = unpack(msg)
        assert result == "direct"


class TestRoundtripDeprecated:
    def test_pack_unpack_roundtrip(self):
        from avp.easy import pack, unpack

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            msg = pack("roundtrip test")
            raw = msg.to_bytes()
            result = unpack(raw)
        assert result == "roundtrip test"
