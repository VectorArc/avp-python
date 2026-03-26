"""Tests for AVP JSON fallback messages."""



from avp.fallback import JSONMessage


# --- JSONMessage ---


def test_json_message_roundtrip_dict():
    msg = JSONMessage(
        session_id="sess-1",
        source_agent_id="alice",
        target_agent_id="bob",
        content="Hello, Bob!",
        extra={"priority": "high"},
    )
    d = msg.to_dict()
    restored = JSONMessage.from_dict(d)

    assert restored.session_id == "sess-1"
    assert restored.source_agent_id == "alice"
    assert restored.target_agent_id == "bob"
    assert restored.content == "Hello, Bob!"
    assert restored.extra["priority"] == "high"


def test_json_message_roundtrip_json():
    msg = JSONMessage(
        session_id="sess-2",
        source_agent_id="a",
        content="Test message",
    )
    json_str = msg.to_json()
    restored = JSONMessage.from_json(json_str)

    assert restored.session_id == "sess-2"
    assert restored.content == "Test message"


def test_json_message_defaults():
    msg = JSONMessage()
    from avp.types import AVP_VERSION_STRING
    assert msg.avp_version == AVP_VERSION_STRING
    assert msg.session_id == ""
    assert msg.content == ""
    assert msg.extra == {}


def test_json_message_no_extra_in_dict():
    msg = JSONMessage(content="no extra")
    d = msg.to_dict()
    assert "extra" not in d


