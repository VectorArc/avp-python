"""Tests for AVP engine connectors (requires torch + transformers)."""

import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

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


# ---------------------------------------------------------------------------
# Multi-embedding / collect_hidden_states tests
# ---------------------------------------------------------------------------


def test_generate_latent_steps_collect_hidden_states(tiny_tied_connector):
    """collect_hidden_states=True returns (past_kv, tensor[1+steps, D])."""
    import torch

    conn = tiny_tied_connector
    input_ids = torch.tensor([[10, 20, 30]], device=conn.device)
    latent_steps = 3

    result = conn.generate_latent_steps(
        input_ids, latent_steps=latent_steps, collect_hidden_states=True,
    )

    assert isinstance(result, tuple), "Should return a tuple when collect_hidden_states=True"
    past_kv, hidden_states = result

    # past_kv should be a valid cache
    assert past_kv is not None

    # hidden_states: [1 + latent_steps, D] — initial + one per step
    assert hidden_states.dim() == 2
    assert hidden_states.shape[0] == 1 + latent_steps
    assert hidden_states.shape[1] == conn._identity.hidden_dim


def test_generate_latent_steps_no_collect_backward_compat(tiny_tied_connector):
    """Default (collect_hidden_states=False) returns just past_kv, not a tuple."""
    import torch

    conn = tiny_tied_connector
    input_ids = torch.tensor([[10, 20, 30]], device=conn.device)

    result = conn.generate_latent_steps(input_ids, latent_steps=3)

    # Should NOT be a tuple — backward compatible
    assert not isinstance(result, tuple), "Default should return just past_kv"


def test_project_hidden_for_cross_model_batch(tiny_tied_connector):
    """project_hidden_for_cross_model handles [N, D] input (batch projection)."""
    import torch
    from avp.rosetta.project import vocabulary_mediated_projection

    conn = tiny_tied_connector
    D = conn._identity.hidden_dim

    # Simulate N hidden states
    N = 4
    hidden_batch = torch.randn(N, D, device=conn.device)

    # Get weights for manual projection
    _, lm_head_weight = conn.get_embedding_weights()
    input_embed_weight = conn.model.get_input_embeddings().weight

    # vocabulary_mediated_projection should handle [N, D] input
    projected = vocabulary_mediated_projection(
        hidden_batch,
        source_lm_head_weight=lm_head_weight,
        target_embed_weight=input_embed_weight,
    )

    assert projected.shape == (N, D), f"Expected ({N}, {D}), got {projected.shape}"
