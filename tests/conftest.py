"""Shared test fixtures and markers for AVP tests."""

import sys

import pytest

# Check for optional dependencies
HAS_TORCH = False
HAS_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    pass

# Custom markers
requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not installed"
)
requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)


@pytest.fixture
def model_config_dict():
    """A minimal model config dict for testing."""
    return {
        "model_type": "llama",
        "_name_or_path": "meta-llama/Llama-2-7b",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "tie_word_embeddings": False,
    }


@pytest.fixture
def model_config_dict_tied():
    """A model config dict with tied embeddings."""
    return {
        "model_type": "gpt2",
        "_name_or_path": "gpt2",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "head_dim": 64,
        "tie_word_embeddings": True,
    }
