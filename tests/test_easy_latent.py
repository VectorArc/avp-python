"""Tests for the avp.pack()/unpack() latent path (Layer 2).

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
    # think() returns a minimal AVPContext
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


class TestPackLatent:
    def test_pack_with_think_steps(self, _mock_connector):
        from avp.easy import pack

        msg = pack("test problem", model="test-model", think_steps=5)
        assert msg.context is not None
        assert msg.content == "test problem"

    def test_pack_latent_has_context(self, _mock_connector):
        from avp.easy import pack
        from avp.context import AVPContext

        msg = pack("test", model="test-model", think_steps=5)
        assert isinstance(msg.context, AVPContext)
        assert msg.context.model_hash == "test-hash"
        assert msg.context.num_steps == 5

    def test_pack_latent_identity(self, _mock_connector):
        from avp.easy import pack

        msg = pack("test", model="test-model", think_steps=5)
        assert msg.identity is not None
        assert msg.identity["model_hash"] == "test-hash"


class TestUnpackLatent:
    def test_unpack_with_model_generates(self, _mock_connector):
        from avp.easy import pack, unpack

        msg = pack("What is 2+2?", model="test-model", think_steps=5)
        result = unpack(msg, model="test-model")
        assert result == "generated response"
        # Verify generate() was called with the latent context
        _mock_connector.generate.assert_called_once()
        call_kwargs = _mock_connector.generate.call_args
        assert call_kwargs[1]["context"] is not None

    def test_unpack_latent_context_passthrough(self, _mock_connector):
        from avp.easy import pack, unpack

        ctx = pack("shared context", model="test-model", think_steps=5)
        result = unpack("follow up", model="test-model", context=ctx)
        assert result == "generated response"
        # Verify context was passed through
        call_kwargs = _mock_connector.generate.call_args
        assert call_kwargs[1]["context"] is ctx.context


class TestClearCache:
    def test_clear_cache_frees_connectors(self, _mock_connector):
        from avp.easy import _connector_cache, _identity_cache, clear_cache, pack

        pack("test", model="test-model", think_steps=5)
        # Caches may have entries (though we patched, identity cache may be empty)
        clear_cache()
        assert len(_connector_cache) == 0
        assert len(_identity_cache) == 0
