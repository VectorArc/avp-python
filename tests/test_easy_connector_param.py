"""Tests for passing EngineConnector instances via model= in avp.think() and avp.generate().

Verifies that passing an EngineConnector instance to model= uses it directly,
while passing a string auto-creates a HuggingFaceConnector via the cache.
"""

from unittest.mock import MagicMock

import pytest

from avp.connectors.base import EngineConnector
from avp.context import AVPContext
from avp.errors import ConfigurationError
from avp.results import GenerateResult, ThinkResult
from avp.types import ModelIdentity


@pytest.fixture
def mock_context():
    """A minimal AVPContext for testing."""
    return AVPContext(
        past_key_values=MagicMock(),
        model_hash="test-hash",
        num_steps=5,
        seq_len=20,
        model_family="qwen2",
        hidden_dim=64,
        num_layers=2,
    )


@pytest.fixture
def mock_connector(mock_context):
    """A mock EngineConnector that returns canned results."""
    connector = MagicMock(spec=EngineConnector)
    connector.think.return_value = mock_context
    connector.generate.return_value = "generated text"
    connector.can_think = True
    connector.get_model_identity.return_value = ModelIdentity(
        model_id="qwen2.5:7b", hidden_dim=64, num_layers=2,
    )
    return connector


class TestThinkWithConnector:
    def test_uses_provided_connector(self, mock_connector):
        from avp.easy import think

        result = think("Analyze this", model=mock_connector)
        assert isinstance(result, ThinkResult)
        mock_connector.think.assert_called_once()
        # Prompt is the first positional arg
        assert mock_connector.think.call_args[0][0] == "Analyze this"

    def test_model_label_from_connector(self, mock_connector):
        from avp.easy import think

        result = think("test", model=mock_connector, collect_metrics=True)
        assert result.metrics.model == "qwen2.5:7b"

    def test_requires_model(self):
        from avp.easy import think

        with pytest.raises(ConfigurationError, match="model="):
            think("test")

    def test_rejects_connector_without_can_think(self):
        from avp.easy import think

        no_think = MagicMock(spec=EngineConnector)
        no_think.can_think = False
        with pytest.raises(ConfigurationError, match="does not support latent thinking"):
            think("test", model=no_think)

    def test_rejects_invalid_model_type(self):
        from avp.easy import think

        with pytest.raises(ConfigurationError, match="model= must be"):
            think("test", model=12345)

    def test_steps_passed_through(self, mock_connector):
        from avp.easy import think

        think("test", model=mock_connector, steps=10)
        assert mock_connector.think.call_args[1]["steps"] == 10

    def test_context_passed_through(self, mock_connector, mock_context):
        from avp.easy import think

        think("test", model=mock_connector, context=mock_context)
        assert mock_connector.think.call_args[1]["context"] is mock_context


class TestGenerateWithConnector:
    def test_uses_provided_connector(self, mock_connector):
        from avp.easy import generate

        result = generate("Solve this", model=mock_connector)
        assert isinstance(result, GenerateResult)
        assert str(result) == "generated text"

    def test_requires_model(self):
        from avp.easy import generate

        with pytest.raises(ConfigurationError, match="model="):
            generate("test")

    def test_with_context(self, mock_connector, mock_context):
        from avp.easy import generate

        result = generate("test", model=mock_connector, context=mock_context)
        assert isinstance(result, GenerateResult)

    def test_metrics_label_from_connector(self, mock_connector):
        from avp.easy import generate

        result = generate("test", model=mock_connector, collect_metrics=True)
        assert result.metrics.model == "qwen2.5:7b"

    def test_steps_zero_skips_think(self, mock_connector):
        from avp.easy import generate

        generate("test", model=mock_connector, steps=0)
        mock_connector.think.assert_not_called()
        mock_connector.generate.assert_called_once()

    def test_rejects_no_think_connector_with_steps(self):
        from avp.easy import generate

        no_think = MagicMock(spec=EngineConnector)
        no_think.can_think = False
        no_think.generate.return_value = "text"
        with pytest.raises(ConfigurationError, match="steps=0"):
            generate("test", model=no_think)

    def test_no_think_connector_with_steps_zero(self):
        from avp.easy import generate

        no_think = MagicMock(spec=EngineConnector)
        no_think.can_think = False
        no_think.generate.return_value = "text-only output"
        result = generate("test", model=no_think, steps=0)
        assert str(result) == "text-only output"


class TestGenerateCrossModelWithConnectors:
    def test_source_and_target_connectors(self, mock_context):
        from avp.easy import generate

        source = MagicMock(spec=EngineConnector)
        source.think.return_value = mock_context
        source.can_think = True
        source.get_model_identity.return_value = ModelIdentity(
            model_id="qwen2.5:7b", hidden_dim=64, num_layers=2,
        )

        target = MagicMock(spec=EngineConnector)
        target.generate.return_value = "cross-model output"
        target.get_model_identity.return_value = ModelIdentity(
            model_id="llama3.2:3b", hidden_dim=48, num_layers=2,
        )

        result = generate(
            "test",
            model=target,
            source_model=source,
            cross_model=True,
        )
        assert str(result) == "cross-model output"
        source.think.assert_called_once()
        target.generate.assert_called_once()
        # Verify source= is passed to target.generate for rosetta
        assert target.generate.call_args[1]["source"] is source

    def test_source_without_cross_model_warns(self, mock_connector):
        from avp.easy import generate

        source = MagicMock(spec=EngineConnector)
        source.can_think = True
        source.get_model_identity.return_value = ModelIdentity(
            model_id="qwen2.5:7b", hidden_dim=64, num_layers=2,
        )

        with pytest.warns(UserWarning, match="experimental"):
            generate("test", model=mock_connector, source_model=source)


class TestBackwardCompatibility:
    """Verify that model= with a string still works (auto-creates HF connector)."""

    def test_think_with_model_string(self, monkeypatch, mock_context):
        from avp import easy

        mock = MagicMock()
        mock.think.return_value = mock_context
        monkeypatch.setattr(easy, "_get_or_create_connector", lambda _: mock)

        result = easy.think("test", model="Qwen/Qwen2.5-7B-Instruct")
        assert isinstance(result, ThinkResult)

    def test_generate_with_model_string(self, monkeypatch, mock_context):
        from avp import easy

        mock = MagicMock()
        mock.think.return_value = mock_context
        mock.generate.return_value = "output"
        monkeypatch.setattr(easy, "_get_or_create_connector", lambda _: mock)

        result = easy.generate("test", model="Qwen/Qwen2.5-7B-Instruct")
        assert isinstance(result, GenerateResult)

    def test_no_deprecation_warning_on_model_string(self, monkeypatch, mock_context):
        """model= with a string should NOT emit a deprecation warning."""
        from avp import easy
        import warnings

        mock = MagicMock()
        mock.think.return_value = mock_context
        mock.generate.return_value = "output"
        monkeypatch.setattr(easy, "_get_or_create_connector", lambda _: mock)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # This should NOT raise — no deprecation warning on model= anymore
            easy.generate("test", model="Qwen/Qwen2.5-7B-Instruct")
