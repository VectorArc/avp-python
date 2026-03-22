"""Tests for AVP vLLM model plugin (mock-based, no vLLM or GPU required)."""

import os

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class MockConfig:
    """Mock model config mimicking HuggingFace/vLLM config."""

    def __init__(self, tie_word_embeddings=True, hidden_size=64, vocab_size=100):
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def to_dict(self):
        return {
            "tie_word_embeddings": self.tie_word_embeddings,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
        }


class MockEmbedding:
    """Mock embedding layer with weight tensor."""

    def __init__(self, vocab_size, hidden_size):
        self.weight = torch.randn(vocab_size, hidden_size)


class MockModel:
    """Mock inner model (the transformer)."""

    def __init__(self, hidden_size=64, vocab_size=100):
        self.embed_tokens = MockEmbedding(vocab_size, hidden_size)


class MockLMHead:
    """Mock lm_head output layer."""

    def __init__(self, vocab_size, hidden_size):
        self.weight = torch.randn(vocab_size, hidden_size)


def _make_plugin(
    tie_word_embeddings=True,
    hidden_size=64,
    vocab_size=100,
    latent_steps=3,
    forward_fn=None,
):
    """Create an AVPLatentQwen2ForCausalLM in stub mode for testing."""
    os.environ["AVP_LATENT_STEPS"] = str(latent_steps)

    from avp.connectors.vllm_model_plugin import AVPLatentQwen2ForCausalLM

    config = MockConfig(tie_word_embeddings, hidden_size, vocab_size)
    model = MockModel(hidden_size, vocab_size)
    lm_head = MockLMHead(vocab_size, hidden_size) if not tie_word_embeddings else None

    plugin = AVPLatentQwen2ForCausalLM(
        config=config,
        model=model,
        lm_head=lm_head,
    )

    # Set up a mock forward function
    call_count = {"n": 0}

    def default_forward(input_ids=None, positions=None, inputs_embeds=None, **kw):
        call_count["n"] += 1
        if inputs_embeds is not None:
            batch = inputs_embeds.shape[0]
            return torch.randn(batch, 1, hidden_size)
        elif input_ids is not None:
            batch = input_ids.shape[0] if input_ids.dim() > 1 else 1
            seq = input_ids.shape[-1] if input_ids.dim() > 0 else 1
            return torch.randn(batch, seq, hidden_size)
        return torch.randn(1, 1, hidden_size)

    plugin._mock_forward = forward_fn or default_forward
    plugin._call_count = call_count
    return plugin


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShouldThink:
    def test_returns_true_for_prefill(self):
        plugin = _make_plugin(latent_steps=3)
        assert plugin._should_think(seq_len=10) is True

    def test_returns_false_for_decode(self):
        plugin = _make_plugin(latent_steps=3)
        assert plugin._should_think(seq_len=1) is False

    def test_returns_false_when_steps_zero(self):
        plugin = _make_plugin(latent_steps=0)
        assert plugin._should_think(seq_len=10) is False

    def test_returns_false_at_threshold(self):
        plugin = _make_plugin(latent_steps=3)
        # threshold is 2, so seq_len=2 should NOT think
        assert plugin._should_think(seq_len=2) is False

    def test_returns_true_above_threshold(self):
        plugin = _make_plugin(latent_steps=3)
        assert plugin._should_think(seq_len=3) is True


class TestProjectHidden:
    def test_tied_model_uses_softmax_projection(self):
        import numpy as np
        plugin = _make_plugin(tie_word_embeddings=True, hidden_size=64)
        plugin._setup_projection()

        hidden = torch.randn(1, 64)
        projected = plugin._project_hidden(hidden)

        assert projected.shape == (1, 64)
        # Should be a weighted average of embeddings, not just normalized hidden
        hidden_np = hidden.detach().float().numpy()
        proj_np = projected if isinstance(projected, np.ndarray) else projected.detach().float().numpy()
        assert not np.allclose(proj_np, hidden_np, atol=0.1)

    def test_untied_model_uses_realignment(self):
        plugin = _make_plugin(tie_word_embeddings=False, hidden_size=64)
        plugin._setup_projection()

        hidden = torch.randn(1, 64)
        projected = plugin._project_hidden(hidden)

        assert projected.shape == (1, 64)

    def test_projected_has_correct_shape(self):
        plugin = _make_plugin(hidden_size=128)
        plugin._setup_projection()

        hidden = torch.randn(1, 128)
        projected = plugin._project_hidden(hidden)

        assert projected.shape == (1, 128)


class TestForward:
    def test_no_latent_steps_single_forward(self):
        """Forward with latent_steps=0 behaves like base model (1 forward call)."""
        plugin = _make_plugin(latent_steps=0)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        positions = torch.tensor([0, 1, 2, 3, 4])

        result = plugin.forward(input_ids, positions)

        assert result is not None
        assert plugin._call_count["n"] == 1

    def test_latent_steps_correct_call_count(self):
        """Forward with N latent steps calls forward N+1 times (1 initial + N steps)."""
        plugin = _make_plugin(latent_steps=3, hidden_size=64)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        positions = torch.tensor([0, 1, 2, 3, 4])

        result = plugin.forward(input_ids, positions)

        assert result is not None
        # 1 initial forward + 3 latent steps = 4 total
        assert plugin._call_count["n"] == 4

    def test_decode_skips_latent_steps(self):
        """During decode (seq_len=1), latent steps are skipped."""
        plugin = _make_plugin(latent_steps=3)

        input_ids = torch.tensor([[42]])
        positions = torch.tensor([10])

        result = plugin.forward(input_ids, positions)

        assert result is not None
        assert plugin._call_count["n"] == 1

    def test_nan_triggers_early_exit(self):
        """NaN in hidden state stops latent loop early."""
        call_count = {"n": 0}

        def forward_with_nan(input_ids=None, inputs_embeds=None, **kw):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                # Return NaN after first latent step
                return torch.full((1, 1, 64), float("nan"))
            if input_ids is not None:
                return torch.randn(1, input_ids.shape[-1], 64)
            return torch.randn(1, 1, 64)

        plugin = _make_plugin(latent_steps=5, hidden_size=64, forward_fn=forward_with_nan)
        plugin._call_count = call_count

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        positions = torch.tensor([0, 1, 2, 3, 4])

        plugin.forward(input_ids, positions)

        # Should exit early: 1 initial + 1 step (NaN detected) = 2
        assert call_count["n"] == 2

    def test_output_shape_preserved(self):
        """Output shape matches what the model produces."""
        plugin = _make_plugin(latent_steps=2, hidden_size=64)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        positions = torch.tensor([0, 1, 2, 3, 4])

        result = plugin.forward(input_ids, positions)

        # Final output is from a latent step: [batch, 1, hidden]
        assert result.dim() == 3
        assert result.shape[0] == 1
        assert result.shape[2] == 64


class TestSetupProjection:
    def test_tied_model_caches_embed_weight(self):
        plugin = _make_plugin(tie_word_embeddings=True, hidden_size=64)
        plugin._setup_projection()

        assert plugin._projection_ready is True
        assert plugin._is_tied is True
        assert plugin._embed_weight is not None
        assert plugin._target_norm is not None

    def test_untied_model_computes_realignment(self):
        plugin = _make_plugin(tie_word_embeddings=False, hidden_size=64)
        plugin._setup_projection()

        assert plugin._projection_ready is True
        assert plugin._is_tied is False
        assert plugin._w_realign is not None
        assert plugin._target_norm is not None


class TestRegistration:
    def test_register_reads_env_var(self, monkeypatch):
        """Plugin reads AVP_LATENT_STEPS from environment."""
        monkeypatch.setenv("AVP_LATENT_STEPS", "7")
        plugin = _make_plugin(latent_steps=7)
        assert plugin._num_latent_steps == 7

    def test_default_latent_steps(self, monkeypatch):
        """Default latent steps is 10 when env var is not set."""
        monkeypatch.delenv("AVP_LATENT_STEPS", raising=False)

        from avp.connectors.vllm_model_plugin import _DEFAULT_LATENT_STEPS

        assert _DEFAULT_LATENT_STEPS == 20
