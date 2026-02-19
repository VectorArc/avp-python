"""Tests for AVP realignment matrix computation (requires torch)."""

import tempfile
from pathlib import Path

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture
def tiny_model():
    """Create a tiny mock model with untied embeddings (hidden_dim=8, vocab=16)."""
    import torch
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 8)
            self.lm_head = nn.Linear(8, 16, bias=False)
            # Make weights different (untied)
            nn.init.normal_(self.embed_tokens.weight, std=0.1)
            nn.init.normal_(self.lm_head.weight, std=0.1)
            self.config = type("Config", (), {
                "tie_word_embeddings": False,
                "to_dict": lambda self_: {
                    "model_type": "tiny",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "tie_word_embeddings": False,
                },
            })()

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def parameters(self):
            yield from super().parameters()

    return TinyModel()


@pytest.fixture
def tiny_model_tied():
    """Create a tiny mock model with tied embeddings."""
    import torch
    import torch.nn as nn

    class TinyModelTied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 8)
            self.lm_head = nn.Linear(8, 16, bias=False)
            # Tie weights
            self.lm_head.weight = self.embed_tokens.weight
            self.config = type("Config", (), {
                "tie_word_embeddings": True,
                "to_dict": lambda self_: {
                    "model_type": "tiny",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "tie_word_embeddings": True,
                },
            })()

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def parameters(self):
            yield from super().parameters()

    return TinyModelTied()


def test_needs_realignment_untied(tiny_model):
    from avp.realign import needs_realignment
    assert needs_realignment(tiny_model) is True


def test_needs_realignment_tied(tiny_model_tied):
    from avp.realign import needs_realignment
    assert needs_realignment(tiny_model_tied) is False


def test_needs_realignment_from_dict():
    from avp.realign import needs_realignment
    assert needs_realignment({"tie_word_embeddings": False}) is True
    assert needs_realignment({"tie_word_embeddings": True}) is False
    assert needs_realignment({}) is True  # Default: needs realignment


def test_compute_realignment_matrix(tiny_model):
    import torch
    from avp.realign import compute_realignment_matrix

    w_realign, target_norm = compute_realignment_matrix(tiny_model)

    assert w_realign.shape == (8, 8)  # hidden_dim x hidden_dim
    assert w_realign.dtype == torch.float32
    assert target_norm.ndim == 0  # scalar
    assert target_norm.item() > 0


def test_apply_realignment(tiny_model):
    import torch
    from avp.realign import apply_realignment, compute_realignment_matrix

    w_realign, target_norm = compute_realignment_matrix(tiny_model)

    # Apply to a batch of hidden states
    hidden = torch.randn(2, 8)  # [batch, hidden_dim]
    aligned = apply_realignment(hidden, w_realign, target_norm)

    assert aligned.shape == hidden.shape
    assert aligned.dtype == hidden.dtype

    # Aligned vectors should have norm close to target_norm
    norms = aligned.norm(dim=-1)
    for n in norms:
        assert abs(n.item() - target_norm.item()) < 0.01


def test_save_load_realignment_matrix(tiny_model):
    import torch
    from avp.realign import (
        compute_realignment_matrix,
        load_realignment_matrix,
        save_realignment_matrix,
    )

    w_realign, target_norm = compute_realignment_matrix(tiny_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        save_realignment_matrix(w_realign, target_norm, "test-hash", cache_dir=cache_dir)

        loaded = load_realignment_matrix("test-hash", cache_dir=cache_dir)
        assert loaded is not None
        w_loaded, norm_loaded = loaded

        torch.testing.assert_close(w_loaded, w_realign.cpu())
        torch.testing.assert_close(norm_loaded, target_norm.cpu())


def test_load_nonexistent_returns_none():
    from avp.realign import load_realignment_matrix

    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_realignment_matrix("nonexistent", cache_dir=Path(tmpdir))
        assert result is None


def test_get_or_compute(tiny_model):
    import torch
    from avp.realign import get_or_compute_realignment

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # First call: computes and saves
        w1, n1 = get_or_compute_realignment(
            tiny_model, "test-model", cache_dir=cache_dir
        )
        assert w1.shape == (8, 8)

        # Second call: loads from cache
        w2, n2 = get_or_compute_realignment(
            tiny_model, "test-model", cache_dir=cache_dir
        )
        torch.testing.assert_close(w1.cpu(), w2.cpu())
