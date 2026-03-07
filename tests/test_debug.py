"""Tests for AVP debug mode (DebugConfig, TransferDiagnostics, inspect, compare)."""

import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import avp
from avp.metrics import (
    DebugConfig,
    GenerateMetrics,
    ThinkMetrics,
    TransferDiagnostics,
)


# ---------------------------------------------------------------------------
# DebugConfig dataclass
# ---------------------------------------------------------------------------


class TestDebugConfig:
    def test_defaults(self):
        cfg = DebugConfig()
        assert cfg.health is True
        assert cfg.compare is False
        assert cfg.step_tokens == 0
        assert cfg.step_snapshots == 0

    def test_compare_mode(self):
        cfg = DebugConfig(compare=True)
        assert cfg.health is True
        assert cfg.compare is True

    def test_future_tiers(self):
        cfg = DebugConfig(step_tokens=5, step_snapshots=4)
        assert cfg.step_tokens == 5
        assert cfg.step_snapshots == 4

    def test_all_tiers(self):
        cfg = DebugConfig(compare=True, step_tokens=10, step_snapshots=3)
        assert cfg.health is True
        assert cfg.compare is True
        assert cfg.step_tokens == 10
        assert cfg.step_snapshots == 3


def test_debug_config_importable():
    assert avp.DebugConfig is DebugConfig


# ---------------------------------------------------------------------------
# TransferDiagnostics dataclass
# ---------------------------------------------------------------------------


class TestTransferDiagnosticsDefaults:
    def test_defaults(self):
        d = TransferDiagnostics()
        assert d.output_empty is False
        assert d.has_nan is False
        assert d.has_inf is False
        assert d.output_length == 0
        assert d.transfer_mode == ""
        assert d.source_model is None
        assert d.target_model is None
        assert d.prompt_tokens == 0
        assert d.quality_gate_passed is None
        assert d.quality_gate_reason == ""
        assert d.projection_method == ""
        assert d.hidden_state_norm is None
        assert d.nearest_cos_sim is None
        assert d.norm_trajectory is None
        assert d.text_baseline_output is None
        assert d.text_overlap is None
        assert d.warnings == []

    def test_healthy_by_default(self):
        d = TransferDiagnostics()
        assert d.healthy is True

    def test_unhealthy_empty_output(self):
        d = TransferDiagnostics(output_empty=True)
        assert d.healthy is False

    def test_unhealthy_nan(self):
        d = TransferDiagnostics(has_nan=True)
        assert d.healthy is False

    def test_unhealthy_inf(self):
        d = TransferDiagnostics(has_inf=True)
        assert d.healthy is False

    def test_unhealthy_all(self):
        d = TransferDiagnostics(output_empty=True, has_nan=True, has_inf=True)
        assert d.healthy is False


class TestTransferDiagnosticsSummary:
    def test_ok_summary(self):
        d = TransferDiagnostics()
        assert d.summary().startswith("OK")

    def test_unhealthy_summary(self):
        d = TransferDiagnostics(output_empty=True, has_nan=True)
        s = d.summary()
        assert "UNHEALTHY" in s
        assert "EMPTY_OUTPUT" in s
        assert "NaN" in s

    def test_mode_in_summary(self):
        d = TransferDiagnostics(transfer_mode="latent")
        assert "mode=latent" in d.summary()

    def test_projection_method_in_summary(self):
        d = TransferDiagnostics(projection_method="VOCAB_OVERLAP")
        assert "projection=VOCAB_OVERLAP" in d.summary()

    def test_cos_sim_in_summary(self):
        d = TransferDiagnostics(nearest_cos_sim=0.876)
        assert "cos_sim=0.876" in d.summary()

    def test_norm_trajectory_in_summary(self):
        d = TransferDiagnostics(norm_trajectory=[10.5, 12.3, 11.8])
        s = d.summary()
        assert "norms=[10.5..11.8]" in s

    def test_overlap_in_summary(self):
        d = TransferDiagnostics(text_overlap=0.45)
        assert "overlap=45%" in d.summary()

    def test_warnings_count_in_summary(self):
        d = TransferDiagnostics(warnings=["w1", "w2"])
        assert "warnings=2" in d.summary()


# ---------------------------------------------------------------------------
# Diagnostics field on ThinkMetrics / GenerateMetrics
# ---------------------------------------------------------------------------


class TestMetricsDiagnosticsField:
    def test_think_metrics_diagnostics_default_none(self):
        m = ThinkMetrics()
        assert m.diagnostics is None

    def test_generate_metrics_diagnostics_default_none(self):
        m = GenerateMetrics()
        assert m.diagnostics is None

    def test_think_metrics_with_diagnostics(self):
        diag = TransferDiagnostics(transfer_mode="latent")
        m = ThinkMetrics(diagnostics=diag)
        assert m.diagnostics is diag
        assert m.diagnostics.transfer_mode == "latent"

    def test_generate_metrics_with_diagnostics(self):
        diag = TransferDiagnostics(output_empty=True)
        m = GenerateMetrics(diagnostics=diag)
        assert m.diagnostics.output_empty is True
        assert m.diagnostics.healthy is False


# ---------------------------------------------------------------------------
# TransferDiagnostics importable from avp
# ---------------------------------------------------------------------------


def test_transfer_diagnostics_importable():
    assert avp.TransferDiagnostics is TransferDiagnostics


# ---------------------------------------------------------------------------
# inspect()
# ---------------------------------------------------------------------------


class TestInspect:
    def _make_avp_binary(self, model_id="test-model", hidden_dim=768, num_layers=12):
        """Create minimal valid AVP binary."""
        metadata = avp.AVPMetadata(
            model_id=model_id,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            payload_type=avp.PayloadType.HIDDEN_STATE,
            dtype=avp.DataType.FLOAT32,
            tensor_shape=(1, hidden_dim),
            mode=avp.CommunicationMode.LATENT,
        )
        payload = np.zeros((1, hidden_dim), dtype=np.float32).tobytes()
        return avp.encode(payload, metadata)

    def test_inspect_returns_dict(self):
        data = self._make_avp_binary()
        result = avp.inspect(data)
        assert isinstance(result, dict)

    def test_inspect_fields(self):
        data = self._make_avp_binary(model_id="Qwen/test", hidden_dim=3584, num_layers=28)
        result = avp.inspect(data)
        assert result["version"] == 1
        assert result["model_id"] == "Qwen/test"
        assert result["hidden_dim"] == 3584
        assert result["num_layers"] == 28
        assert result["payload_type"] == "HIDDEN_STATE"
        assert result["mode"] == "LATENT"
        assert result["dtype"] == "FLOAT32"
        assert isinstance(result["raw_size"], int)
        assert result["raw_size"] > 0

    def test_inspect_flags(self):
        data = self._make_avp_binary()
        result = avp.inspect(data)
        assert "compressed" in result
        assert "hybrid" in result
        assert "has_map" in result
        assert "kv_cache" in result
        assert result["compressed"] is False
        assert result["kv_cache"] is False

    def test_inspect_invalid_magic(self):
        data = b"\x00\x00" + b"\x00" * 20
        with pytest.raises(avp.errors.InvalidMagicError):
            avp.inspect(data)

    def test_inspect_truncated(self):
        with pytest.raises(avp.errors.DecodeError):
            avp.inspect(b"\x41\x56")

    def test_inspect_in_all(self):
        assert "inspect" in avp.__all__


# ---------------------------------------------------------------------------
# _compute_word_overlap
# ---------------------------------------------------------------------------


class TestWordOverlap:
    def test_identical(self):
        from avp.easy import _compute_word_overlap
        assert _compute_word_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        from avp.easy import _compute_word_overlap
        assert _compute_word_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        from avp.easy import _compute_word_overlap
        result = _compute_word_overlap("the cat sat", "the dog sat on mat")
        # shared: {"the", "sat"}, max(3, 5) = 5, overlap = 2/5 = 0.4
        assert abs(result - 0.4) < 1e-6

    def test_both_empty(self):
        from avp.easy import _compute_word_overlap
        assert _compute_word_overlap("", "") == 1.0

    def test_one_empty(self):
        from avp.easy import _compute_word_overlap
        assert _compute_word_overlap("hello", "") == 0.0
        assert _compute_word_overlap("", "hello") == 0.0
