"""Tests for AVP JSON fallback messages."""

import json

import pytest

from avp.fallback import FallbackRequest, JSONMessage


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
    assert msg.avp_version == "0.2.0"
    assert msg.session_id == ""
    assert msg.content == ""
    assert msg.extra == {}


def test_json_message_no_extra_in_dict():
    msg = JSONMessage(content="no extra")
    d = msg.to_dict()
    assert "extra" not in d


# --- FallbackRequest ---


def test_fallback_request_roundtrip_dict():
    req = FallbackRequest(
        session_id="sess-1",
        reason="incompatible models",
        perplexity_score=15.5,
    )
    d = req.to_dict()
    restored = FallbackRequest.from_dict(d)

    assert restored.session_id == "sess-1"
    assert restored.reason == "incompatible models"
    assert restored.perplexity_score == 15.5


def test_fallback_request_roundtrip_json():
    req = FallbackRequest(
        session_id="sess-2",
        reason="high perplexity",
        perplexity_score=25.0,
    )
    json_str = req.to_json()
    restored = FallbackRequest.from_json(json_str)

    assert restored.reason == "high perplexity"
    assert restored.perplexity_score == 25.0


def test_fallback_request_defaults():
    req = FallbackRequest()
    assert req.session_id == ""
    assert req.reason == ""
    assert req.perplexity_score == 0.0
