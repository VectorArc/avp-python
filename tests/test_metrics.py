"""Tests for AVP observability metrics.

All tests are Layer 0 (no GPU, no model loading).
"""

import json
import logging

import avp
from avp.easy import PackedMessage
from avp.metrics import HandshakeMetrics, PackMetrics, UnpackMetrics


# --- Dataclass defaults ---


def test_pack_metrics_defaults():
    m = PackMetrics()
    assert m.layer == 0
    assert m.model is None
    assert m.think_steps == 0
    assert m.has_prior_context is False
    assert m.duration_s == 0.0
    assert m.identity_duration_s == 0.0
    assert m.think_duration_s == 0.0


def test_unpack_metrics_defaults():
    m = UnpackMetrics()
    assert m.input_format == "unknown"
    assert m.has_context is False
    assert m.generated is False
    assert m.duration_s == 0.0
    assert m.decode_duration_s == 0.0
    assert m.generate_duration_s == 0.0


def test_handshake_metrics_defaults():
    m = HandshakeMetrics()
    assert m.resolution_path == ""
    assert m.mode == ""
    assert m.avp_map_id == ""
    assert m.duration_s == 0.0


def test_pack_metrics_field_assignment():
    m = PackMetrics(layer=2, model="test", think_steps=20, duration_s=1.5)
    assert m.layer == 2
    assert m.model == "test"
    assert m.think_steps == 20
    assert m.duration_s == 1.5


# --- pack() with collect_metrics ---


def test_pack_with_metrics_returns_tuple():
    result = avp.pack("hello", collect_metrics=True)
    assert isinstance(result, tuple)
    msg, metrics = result
    assert isinstance(msg, PackedMessage)
    assert isinstance(metrics, PackMetrics)


def test_pack_with_metrics_layer0():
    msg, metrics = avp.pack("hello", collect_metrics=True)
    assert metrics.layer == 0
    assert metrics.model is None
    assert metrics.think_steps == 0
    assert metrics.has_prior_context is False
    assert metrics.duration_s > 0
    assert msg.content == "hello"


def test_pack_without_metrics_returns_packed_message():
    result = avp.pack("hello")
    assert isinstance(result, PackedMessage)
    assert not isinstance(result, tuple)


def test_pack_metrics_has_prior_context():
    prior = PackedMessage(content="prior")
    _, metrics = avp.pack("hello", context=prior, collect_metrics=True)
    assert metrics.has_prior_context is True


# --- unpack() with collect_metrics ---


def test_unpack_text_with_metrics():
    result = avp.unpack("plain text", collect_metrics=True)
    assert isinstance(result, tuple)
    text, metrics = result
    assert text == "plain text"
    assert isinstance(metrics, UnpackMetrics)
    assert metrics.input_format == "text"
    assert metrics.generated is False
    assert metrics.duration_s > 0


def test_unpack_without_metrics_returns_str():
    result = avp.unpack("plain text")
    assert isinstance(result, str)
    assert result == "plain text"


def test_unpack_json_format():
    data = json.dumps({"avp": "0.2", "content": "hello"})
    text, metrics = avp.unpack(data, collect_metrics=True)
    assert text == "hello"
    assert metrics.input_format == "json"


def test_unpack_json_bytes_format():
    data = json.dumps({"avp": "0.2", "content": "hello"}).encode("utf-8")
    text, metrics = avp.unpack(data, collect_metrics=True)
    assert text == "hello"
    assert metrics.input_format == "json"


def test_unpack_packed_message_format():
    msg = PackedMessage(content="hi")
    text, metrics = avp.unpack(msg, collect_metrics=True)
    assert text == "hi"
    assert metrics.input_format == "packed_message"


def test_unpack_text_bytes_format():
    data = b"just bytes"
    text, metrics = avp.unpack(data, collect_metrics=True)
    assert text == "just bytes"
    assert metrics.input_format == "text"


# --- Logging ---


def test_pack_logs_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="avp.easy"):
        avp.pack("hello")
    assert any("pack() layer=0" in r.message for r in caplog.records)


def test_unpack_logs_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="avp.easy"):
        avp.unpack("hello")
    assert any("unpack() format=text" in r.message for r in caplog.records)


# --- Lazy import from avp namespace ---


def test_metrics_importable_from_avp():
    assert avp.PackMetrics is PackMetrics
    assert avp.UnpackMetrics is UnpackMetrics
    assert avp.HandshakeMetrics is HandshakeMetrics
