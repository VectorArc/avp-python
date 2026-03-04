"""Tests for the avp.think() easy API (latent path).

Requires torch + transformers.
"""

import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
    pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed"),
]


@pytest.fixture
def _mock_connector(monkeypatch):
    """Mock connector that returns canned results for think() and generate()."""
    from unittest.mock import MagicMock
    from avp.context import AVPContext
    from avp import easy

    mock = MagicMock()
    mock_cache = MagicMock()
    mock_context = AVPContext(
        past_key_values=mock_cache,
        model_hash="test-hash",
        num_steps=5,
        seq_len=20,
        model_family="gpt2",
        hidden_dim=64,
        num_layers=2,
    )
    mock.think.return_value = mock_context
    mock.generate.return_value = "generated response"

    monkeypatch.setattr(easy, "_get_or_create_connector", lambda _name: mock)
    monkeypatch.setattr(
        easy,
        "_get_local_identity",
        lambda _name: {
            "model_id": "test-model",
            "model_hash": "test-hash",
            "hidden_dim": 64,
            "num_layers": 2,
            "model_family": "gpt2",
        },
    )
    return mock


class TestThink:
    def test_returns_avp_context(self, _mock_connector):
        from avp.context import AVPContext
        from avp.easy import think

        ctx = think("test problem", model="test-model")
        assert isinstance(ctx, AVPContext)
        assert ctx.model_hash == "test-hash"
        assert ctx.num_steps == 5

    def test_default_steps_20(self, _mock_connector):
        from avp.easy import think

        think("test", model="test-model")
        call_kwargs = _mock_connector.think.call_args[1]
        assert call_kwargs["steps"] == 20

    def test_custom_steps(self, _mock_connector):
        from avp.easy import think

        think("test", model="test-model", steps=5)
        call_kwargs = _mock_connector.think.call_args[1]
        assert call_kwargs["steps"] == 5

    def test_with_prior_context(self, _mock_connector):
        from avp.context import AVPContext
        from avp.easy import think

        prior = AVPContext(
            past_key_values=None,
            model_hash="test-hash",
            num_steps=3,
            seq_len=10,
        )
        think("continue", model="test-model", context=prior)
        call_kwargs = _mock_connector.think.call_args[1]
        assert call_kwargs["context"] is prior

    def test_rejects_non_string(self, _mock_connector):
        from avp.easy import think

        with pytest.raises(TypeError, match="prompt must be str"):
            think(123, model="test-model")

    def test_with_metrics(self, _mock_connector):
        from avp.easy import think
        from avp.metrics import ThinkMetrics

        result = think("test", model="test-model", collect_metrics=True)
        assert isinstance(result, tuple)
        ctx, metrics = result
        assert isinstance(metrics, ThinkMetrics)
        assert metrics.model == "test-model"
        assert metrics.steps == 20
        assert metrics.duration_s > 0


class TestThinkImport:
    def test_importable_from_avp(self, _mock_connector):
        import avp

        assert hasattr(avp, "think")
        ctx = avp.think("hello", model="test-model")
        assert ctx is not None


class TestClearCache:
    def test_clear_cache_frees_connectors(self, _mock_connector):
        from avp.easy import _connector_cache, _identity_cache, clear_cache, think

        think("test", model="test-model")
        clear_cache()
        assert len(_connector_cache) == 0
        assert len(_identity_cache) == 0
