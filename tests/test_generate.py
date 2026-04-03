"""Tests for the avp.generate() combined API.

Uses mocked connector — no GPU needed.
"""

import importlib.util
import logging

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

    from avp import easy
    from avp.context import AVPContext

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
    return mock


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestGenerateBasic:
    def test_returns_text(self, _mock_connector):
        from avp.easy import generate

        result = generate("hello", model="test-model")
        assert result == "generated response"

    def test_calls_think_and_generate(self, _mock_connector):
        from avp.easy import generate

        generate("hello", model="test-model", steps=10)
        _mock_connector.think.assert_called_once()
        _mock_connector.generate.assert_called_once()

    def test_steps_default_10(self, _mock_connector):
        from avp.easy import generate

        generate("hello", model="test-model")
        call_args = _mock_connector.think.call_args
        assert call_args[1]["steps"] == 10

    def test_custom_steps(self, _mock_connector):
        from avp.easy import generate

        generate("hello", model="test-model", steps=5)
        call_args = _mock_connector.think.call_args
        assert call_args[1]["steps"] == 5

    def test_zero_steps(self, _mock_connector):
        from avp.easy import generate

        generate("hello", model="test-model", steps=0)
        _mock_connector.think.assert_not_called()
        _mock_connector.generate.assert_called_once()

    def test_passes_generation_params(self, _mock_connector):
        from avp.easy import generate

        generate(
            "hello", model="test-model", steps=0,
            max_new_tokens=128, temperature=0.3,
        )
        call_kwargs = _mock_connector.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 128
        assert call_kwargs["temperature"] == 0.3


# ---------------------------------------------------------------------------
# ContextStore integration
# ---------------------------------------------------------------------------


class TestGenerateStore:
    def test_store_key_stores_context(self, _mock_connector):
        from avp.context import AVPContext
        from avp.context_store import ContextStore
        from avp.easy import generate

        store = ContextStore()
        generate("hello", model="test-model", store=store, store_key="agent-a")

        result = store.get("agent-a")
        assert result is not None
        assert isinstance(result, AVPContext)

    def test_prior_key_retrieves_context(self, _mock_connector):
        from avp.context_store import ContextStore
        from avp.easy import generate

        store = ContextStore()
        # First call: researcher stores context
        generate("research this", model="test-model",
                 store=store, store_key="researcher")

        # Second call: writer retrieves researcher's context
        _mock_connector.think.reset_mock()
        generate("write about it", model="test-model",
                 store=store, store_key="writer", prior_key="researcher")

        # think() should have been called with context from first call
        call_kwargs = _mock_connector.think.call_args[1]
        assert call_kwargs["context"] is not None

    def test_prior_key_missing_passes_none(self, _mock_connector):
        from avp.context_store import ContextStore
        from avp.easy import generate

        store = ContextStore()
        # prior_key doesn't exist — should pass None, not error
        generate("hello", model="test-model",
                 store=store, prior_key="nonexistent")

        call_kwargs = _mock_connector.think.call_args[1]
        assert call_kwargs["context"] is None

    def test_no_store_no_store_key(self, _mock_connector):
        from avp.easy import generate

        # Should work fine without store
        result = generate("hello", model="test-model")
        assert result == "generated response"

    def test_store_and_prior_in_chain(self, _mock_connector):
        """Full 3-agent chain: A → B → C with context passing."""
        from avp.context_store import ContextStore
        from avp.easy import generate

        store = ContextStore()

        generate("step 1", model="test-model",
                 store=store, store_key="agent-a")
        generate("step 2", model="test-model",
                 store=store, store_key="agent-b", prior_key="agent-a")
        generate("step 3", model="test-model",
                 store=store, store_key="agent-c", prior_key="agent-b")

        assert store.active_count == 3
        assert set(store.keys()) == {"agent-a", "agent-b", "agent-c"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestGenerateValidation:
    def test_rejects_non_string_prompt(self, _mock_connector):
        from avp.easy import generate
        from avp.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="prompt must be str"):
            generate(123, model="test-model")

    def test_store_key_without_store_raises(self, _mock_connector):
        from avp.easy import generate
        from avp.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="store_key/prior_key require store="):
            generate("hello", model="test-model", store_key="key")

    def test_prior_key_without_store_raises(self, _mock_connector):
        from avp.easy import generate
        from avp.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="store_key/prior_key require store="):
            generate("hello", model="test-model", prior_key="key")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestGenerateMetrics:
    def test_collect_metrics_on_result(self, _mock_connector):
        from avp.easy import generate
        from avp.metrics import GenerateMetrics
        from avp.results import GenerateResult

        result = generate("hello", model="test-model", collect_metrics=True)
        assert isinstance(result, GenerateResult)
        assert isinstance(result, str)
        assert result == "generated response"
        assert isinstance(result.metrics, GenerateMetrics)

    def test_metrics_fields(self, _mock_connector):
        from avp.context_store import ContextStore
        from avp.easy import generate

        store = ContextStore()
        result = generate(
            "hello", model="test-model", steps=10,
            store=store, store_key="a",
            collect_metrics=True,
        )
        metrics = result.metrics
        assert metrics.model == "test-model"
        assert metrics.steps == 10
        assert metrics.stored is True
        assert metrics.duration_s > 0
        assert metrics.think_duration_s >= 0
        assert metrics.generate_duration_s >= 0

    def test_metrics_no_store(self, _mock_connector):
        from avp.easy import generate

        result = generate("hello", model="test-model", collect_metrics=True)
        assert result.metrics.stored is False
        assert result.metrics.has_prior_context is False

    def test_metrics_with_prior(self, _mock_connector):
        from avp.context_store import ContextStore
        from avp.easy import generate

        store = ContextStore()
        generate("first", model="test-model", store=store, store_key="a")
        result = generate(
            "second", model="test-model",
            store=store, prior_key="a",
            collect_metrics=True,
        )
        assert result.metrics.has_prior_context is True

    def test_without_metrics_returns_str(self, _mock_connector):
        from avp.easy import generate
        from avp.results import GenerateResult

        result = generate("hello", model="test-model")
        assert isinstance(result, str)
        assert isinstance(result, GenerateResult)
        assert result.metrics is None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestGenerateLogging:
    def test_logs_debug(self, _mock_connector, caplog):
        from avp.easy import generate

        with caplog.at_level(logging.DEBUG, logger="avp.easy"):
            generate("hello", model="test-model")

        assert any("generate()" in r.message for r in caplog.records)

    def test_logs_info_timing(self, _mock_connector, caplog):
        from avp.easy import generate

        with caplog.at_level(logging.INFO, logger="avp.easy"):
            generate("hello", model="test-model")

        assert any("think=" in r.message and "generate=" in r.message
                    for r in caplog.records)


# ---------------------------------------------------------------------------
# Import from avp namespace
# ---------------------------------------------------------------------------


class TestGenerateImport:
    def test_importable_from_avp(self, _mock_connector):
        import avp

        assert hasattr(avp, "generate")
        result = avp.generate("hello", model="test-model")
        assert result == "generated response"

    def test_generate_metrics_importable(self):
        import avp
        from avp.metrics import GenerateMetrics

        assert avp.GenerateMetrics is GenerateMetrics
