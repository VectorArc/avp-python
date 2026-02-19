"""Tests for AVP engine connectors (requires torch + transformers)."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
    pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed"),
]


def test_connector_imports():
    """Verify HuggingFaceConnector can be imported."""
    from avp.connectors import HuggingFaceConnector
    from avp.connectors.base import EngineConnector
    assert issubclass(HuggingFaceConnector, EngineConnector)


def test_connector_get_identity_from_config():
    """Test that extract_model_identity works with a config dict."""
    from avp.handshake import extract_model_identity

    config = {
        "model_type": "qwen2",
        "_name_or_path": "Qwen/Qwen2-0.5B",
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
    }
    identity = extract_model_identity(config)
    assert identity.model_family == "qwen2"
    assert identity.hidden_dim == 896
    assert identity.num_layers == 24
    assert identity.num_kv_heads == 2


def test_connector_needs_realignment():
    """Test realignment check without loading a real model."""
    from avp.realign import needs_realignment

    assert needs_realignment({"tie_word_embeddings": False}) is True
    assert needs_realignment({"tie_word_embeddings": True}) is False
