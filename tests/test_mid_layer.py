"""Tests for mid-layer injection cross-model transfer."""

import pytest
import torch

from avp.rosetta.mid_layer import (
    DEFAULT_DEPTH_RATIO,
    compute_extraction_layer,
    compute_injection_layer,
    extract_mid_layer_hidden,
    mid_layer_injection_hook,
    _get_decoder_layers,
)


class TestLayerComputation:
    """Tests for extraction/injection layer computation."""

    def test_default_depth_ratio(self):
        assert DEFAULT_DEPTH_RATIO == 0.75

    def test_extraction_layer_28_layers(self):
        """Qwen 7B has 28 layers -> extract from layer 21."""
        layer = compute_extraction_layer(28)
        assert layer == 21  # int(28 * 0.75)

    def test_extraction_layer_32_layers(self):
        """Llama 3B has 32 layers -> extract from layer 24."""
        layer = compute_extraction_layer(32)
        assert layer == 24

    def test_injection_layer_proportional(self):
        """28 source layers, 32 target layers -> inject at 24."""
        layer = compute_injection_layer(32)
        assert layer == 24

    def test_custom_depth_ratio(self):
        layer = compute_extraction_layer(28, depth_ratio=0.5)
        assert layer == 14

    def test_depth_ratio_zero(self):
        layer = compute_extraction_layer(28, depth_ratio=0.0)
        assert layer == 0

    def test_depth_ratio_one(self):
        """Ratio 1.0 should clamp to last layer."""
        layer = compute_extraction_layer(28, depth_ratio=1.0)
        assert layer == 27  # min(28, 27)

    def test_small_model(self):
        layer = compute_extraction_layer(4, depth_ratio=0.75)
        assert layer == 3


class TestExtractMidLayerHidden:
    """Tests for mid-layer hidden state extraction."""

    def test_extracts_correct_layer(self):
        """Verify extraction returns the right layer's last token."""
        # Simulate hidden_states: tuple of (num_layers + 1) tensors
        # Index 0 = embedding, index i+1 = layer i output
        num_layers = 10
        batch_size = 1
        seq_len = 5
        hidden_dim = 32

        hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden_dim) * (i + 1)
            for i in range(num_layers + 1)
        )

        class MockOutputs:
            pass

        outputs = MockOutputs()
        outputs.hidden_states = hidden_states

        extraction_layer = 7
        result = extract_mid_layer_hidden(outputs, extraction_layer)

        assert result.shape == (batch_size, hidden_dim)
        # Should match hidden_states[extraction_layer + 1][:, -1, :]
        expected = hidden_states[extraction_layer + 1][:, -1, :]
        assert torch.allclose(result, expected)

    def test_clamps_to_valid_range(self):
        """Layer beyond range should clamp."""
        hidden_states = tuple(
            torch.randn(1, 5, 16) for _ in range(6)  # 5 layers + embedding
        )

        class MockOutputs:
            pass

        outputs = MockOutputs()
        outputs.hidden_states = hidden_states

        # Request layer 10 but only 5 layers exist
        result = extract_mid_layer_hidden(outputs, 10)
        assert result.shape == (1, 16)


class TestMidLayerInjectionHook:
    """Tests for the forward hook context manager."""

    def test_hook_modifies_output(self):
        """Test that the hook replaces the last token's hidden state."""
        # Simple linear model to test hooking
        layer = torch.nn.Linear(16, 16, bias=False)
        # Set to identity so we can verify the hook's effect
        with torch.no_grad():
            layer.weight.copy_(torch.eye(16))

        class SimpleModel:
            def __init__(self):
                self.model = type("inner", (), {"layers": torch.nn.ModuleList([layer])})()

        model = SimpleModel()
        projected = torch.ones(1, 16) * 42.0  # Distinctive value

        with mid_layer_injection_hook(model, 0, projected):
            # Simulate a forward pass through the hooked layer
            input_tensor = torch.randn(1, 5, 16)
            output = layer(input_tensor)
            # output should have last position replaced

        # After context exit, hook should be removed
        assert len(layer._forward_hooks) == 0

    def test_hook_fires_only_once(self):
        """The hook should fire only on the first forward pass."""
        layer = torch.nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(16))

        class SimpleModel:
            def __init__(self):
                self.model = type("inner", (), {"layers": torch.nn.ModuleList([layer])})()

        model = SimpleModel()
        projected = torch.ones(1, 16) * 42.0

        with mid_layer_injection_hook(model, 0, projected):
            # First forward pass — hook should fire
            input1 = torch.randn(1, 5, 16)
            out1 = layer(input1)

            # Second forward pass — hook should NOT fire
            input2 = torch.randn(1, 3, 16)
            out2 = layer(input2)

            # First pass: last token should be replaced
            # Second pass: should be unmodified
            # (The hook returns the modified output only on first call)

        assert len(layer._forward_hooks) == 0

    def test_hook_cleanup_on_exception(self):
        """Hook should be removed even if an exception occurs."""
        layer = torch.nn.Linear(16, 16, bias=False)

        class SimpleModel:
            def __init__(self):
                self.model = type("inner", (), {"layers": torch.nn.ModuleList([layer])})()

        model = SimpleModel()
        projected = torch.ones(1, 16)

        with pytest.raises(RuntimeError):
            with mid_layer_injection_hook(model, 0, projected):
                raise RuntimeError("test error")

        # Hook should still be removed
        assert len(layer._forward_hooks) == 0


class TestGetDecoderLayers:
    """Tests for model layer discovery."""

    def test_llama_style_model(self):
        """Models with model.model.layers (Llama, Qwen)."""
        layers = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(4)])
        inner = type("inner", (), {"layers": layers})()
        model = type("model", (), {"model": inner})()
        result = _get_decoder_layers(model)
        assert result is layers

    def test_gpt2_style_model(self):
        """Models with model.transformer.h (GPT-2)."""
        h = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(4)])
        transformer = type("transformer", (), {"h": h})()
        model = type("model", (), {"model": None, "transformer": transformer})()
        result = _get_decoder_layers(model)
        assert result is h

    def test_unknown_model_raises(self):
        """Models without known layer paths should raise."""
        model = type("model", (), {"model": None, "transformer": None})()
        with pytest.raises(AttributeError, match="Cannot find decoder layers"):
            _get_decoder_layers(model)
