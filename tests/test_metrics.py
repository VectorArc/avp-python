"""Tests for AVP observability metrics."""

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


# --- ThinkMetrics importable from avp ---


def test_think_metrics_importable_from_avp():
    assert avp.ThinkMetrics is ThinkMetrics


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
