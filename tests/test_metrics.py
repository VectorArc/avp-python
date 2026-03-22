"""Tests for AVP observability metrics."""

import json
import warnings

import avp
from avp.metrics import GenerateMetrics, ThinkMetrics


# --- Dataclass defaults ---


def test_think_metrics_defaults():
    m = ThinkMetrics()
    assert m.model is None
    assert m.steps == 0
    assert m.has_prior_context is False
    assert m.duration_s == 0.0
    assert m.think_duration_s == 0.0


def test_generate_metrics_defaults():
    m = GenerateMetrics()
    assert m.model is None
    assert m.steps == 0
    assert m.has_prior_context is False
    assert m.stored is False
    assert m.duration_s == 0.0
    assert m.think_duration_s == 0.0
    assert m.generate_duration_s == 0.0


def test_think_metrics_field_assignment():
    m = ThinkMetrics(model="test", steps=20, duration_s=1.5)
    assert m.model == "test"
    assert m.steps == 20
    assert m.duration_s == 1.5


# --- PackMetrics is alias for ThinkMetrics ---


def test_pack_metrics_is_think_metrics():
    from avp.metrics import PackMetrics
    assert PackMetrics is ThinkMetrics


# --- ThinkMetrics importable from avp ---


def test_think_metrics_importable_from_avp():
    assert avp.ThinkMetrics is ThinkMetrics


# --- Deprecated pack/unpack with collect_metrics ---


def test_pack_with_metrics_returns_tuple():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = avp.pack("hello", collect_metrics=True)
    assert isinstance(result, tuple)
    msg, metrics = result
    assert isinstance(metrics, ThinkMetrics)


def test_pack_with_metrics_fields():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        msg, metrics = avp.pack("hello", collect_metrics=True)
    assert metrics.model is None
    assert metrics.steps == 0
    assert metrics.has_prior_context is False
    assert metrics.duration_s > 0
    assert msg.content == "hello"


def test_unpack_text_with_metrics():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from avp.metrics import UnpackMetrics
        result = avp.unpack("plain text", collect_metrics=True)
    assert isinstance(result, tuple)
    text, metrics = result
    assert text == "plain text"
    assert isinstance(metrics, UnpackMetrics)
    assert metrics.input_format == "text"
    assert metrics.generated is False


def test_unpack_json_format():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        data = json.dumps({"avp": "0.2", "content": "hello"})
        text, metrics = avp.unpack(data, collect_metrics=True)
    assert text == "hello"
    assert metrics.input_format == "json"


# --- TransferDiagnostics on metrics ---


def test_think_metrics_diagnostics_field():
    m = ThinkMetrics()
    assert m.diagnostics is None

    from avp.metrics import TransferDiagnostics
    d = TransferDiagnostics(transfer_mode="latent")
    m2 = ThinkMetrics(diagnostics=d)
    assert m2.diagnostics is not None
    assert m2.diagnostics.transfer_mode == "latent"


def test_generate_metrics_diagnostics_field():
    m = GenerateMetrics()
    assert m.diagnostics is None

    from avp.metrics import TransferDiagnostics
    d = TransferDiagnostics(output_empty=True)
    m2 = GenerateMetrics(diagnostics=d)
    assert m2.diagnostics is not None
    assert m2.diagnostics.healthy is False


def test_transfer_diagnostics_importable_from_avp():
    from avp.metrics import TransferDiagnostics
    assert avp.TransferDiagnostics is TransferDiagnostics
