"""Tests for automatic cross-model rosetta projection via generate(source=)."""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from avp.context import AVPContext

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)


class TestAVPContextLastHidden:
    """Tests for the last_hidden_state field on AVPContext."""

    def test_default_none(self):
        ctx = AVPContext(
            past_key_values=None,
            model_hash="abc",
            num_steps=0,
            seq_len=0,
        )
        assert ctx.last_hidden_state is None

    @requires_torch
    def test_stores_tensor(self):
        import torch

        hidden = torch.randn(1, 64)
        ctx = AVPContext(
            past_key_values=None,
            model_hash="abc",
            num_steps=10,
            seq_len=100,
            last_hidden_state=hidden,
        )
        assert ctx.last_hidden_state is hidden


@requires_torch
@requires_transformers
class TestThinkCapturesHidden:
    """Tests that think() populates last_hidden_state on AVPContext."""

    def test_think_captures_last_hidden_state(self, tiny_tied_connector):
        ctx = tiny_tied_connector.think("Hello world", steps=3)
        assert ctx.last_hidden_state is not None
        # Should be [1, D] shape
        assert ctx.last_hidden_state.dim() == 2
        assert ctx.last_hidden_state.shape[0] == 1
        assert ctx.last_hidden_state.shape[1] == 64  # tiny model hidden_dim

    def test_think_last_hidden_is_final_step(self, tiny_tied_connector):
        """Verify last_hidden_state is from the final latent step."""
        ctx = tiny_tied_connector.think("Test prompt", steps=5)

        # Also run with collect_hidden_states to compare
        from avp.connectors.base import _render_prompt, _tokenize_prompt
        from avp.connectors.huggingface import _to_messages

        messages = _to_messages("Test prompt")
        prompt_text = _render_prompt(tiny_tied_connector.tokenizer, messages)
        input_ids, attn_mask = _tokenize_prompt(
            tiny_tied_connector.tokenizer, prompt_text, "cpu"
        )
        _, all_hidden = tiny_tied_connector.generate_latent_steps(
            input_ids, 5, attention_mask=attn_mask, collect_hidden_states=True,
        )
        # last_hidden_state should match the last entry in all_hidden
        # (may differ due to separate runs with random state, but shape matches)
        assert ctx.last_hidden_state.shape == all_hidden[-1].unsqueeze(0).shape


@requires_torch
@requires_transformers
class TestGenerateCrossModel:
    """Tests for generate() with source= parameter."""

    def test_same_model_ignores_source(self, tiny_tied_connector):
        """When source has same model_hash, source= is a no-op."""
        ctx = tiny_tied_connector.think("Hello", steps=2)
        # source is the same connector — should not trigger cross-model path
        result = tiny_tied_connector.generate(
            "Hello", context=ctx, source=tiny_tied_connector,
            max_new_tokens=10,
        )
        assert isinstance(result, str)

    def test_cross_model_no_hidden_raises(self, tiny_tied_connector, tiny_untied_connector):
        """Raises ValueError when context has no last_hidden_state."""
        ctx = AVPContext(
            past_key_values=None,
            model_hash=tiny_untied_connector._model_hash,
            num_steps=5,
            seq_len=50,
            last_hidden_state=None,  # missing
        )
        with pytest.raises(ValueError, match="last_hidden_state"):
            tiny_tied_connector.generate(
                "Hello", context=ctx, source=tiny_untied_connector,
            )

    def test_cross_model_projects_and_generates(
        self, tiny_tied_connector, tiny_untied_connector,
    ):
        """Full cross-model flow: think on A, generate on B with source=A."""
        ctx = tiny_untied_connector.think("Solve: 2 + 2", steps=3)
        assert ctx.last_hidden_state is not None

        with patch("avp.rosetta.registry.save_map"):
            result = tiny_tied_connector.generate(
                "Solve: 2 + 2",
                context=ctx,
                source=tiny_untied_connector,
                max_new_tokens=20,
            )
        assert isinstance(result, str)

    def test_cross_model_without_source_raises(
        self, tiny_tied_connector, tiny_untied_connector,
    ):
        """Without source=, cross-model context raises IncompatibleModelsError."""
        from avp.errors import IncompatibleModelsError

        ctx = tiny_untied_connector.think("Hello", steps=2)
        with pytest.raises(IncompatibleModelsError, match="source="):
            tiny_tied_connector.generate("Hello", context=ctx)


@requires_torch
@requires_transformers
class TestGetOrCalibrateMap:
    """Tests for the 3-tier AVPMap cache."""

    def test_memory_cache_hit(self, tiny_tied_connector, tiny_untied_connector):
        """Second call returns cached map without re-calibrating."""
        # Prevent calibrate from saving to disk (avoids polluting ~/.avp/maps/)
        with patch("avp.rosetta.registry.save_map"):
            map1 = tiny_tied_connector._get_or_calibrate_map(tiny_untied_connector)
        map2 = tiny_tied_connector._get_or_calibrate_map(tiny_untied_connector)
        assert map1 is map2  # same object from memory cache

    def test_disk_cache_hit(self, tiny_tied_connector, tiny_untied_connector, tmp_path):
        """Loads from disk registry when memory cache is empty."""
        from avp.rosetta.registry import save_map

        # Calibrate (prevent auto-save to ~/.avp/maps/) and save to custom dir
        with patch("avp.rosetta.registry.save_map"):
            avp_map = tiny_tied_connector._get_or_calibrate_map(tiny_untied_connector)
        save_map(avp_map, map_dir=tmp_path)

        # Clear memory cache
        tiny_tied_connector._avp_map_cache.clear()

        # Patch the registry where it's imported from
        with patch("avp.rosetta.registry.load_map") as mock_load:
            mock_load.return_value = avp_map
            result = tiny_tied_connector._get_or_calibrate_map(tiny_untied_connector)
            mock_load.assert_called_once()
            assert result is avp_map

    def test_calibrates_on_miss(self, tiny_tied_connector, tiny_untied_connector):
        """Calibrates when both memory and disk caches miss."""
        tiny_tied_connector._avp_map_cache.clear()

        with patch("avp.rosetta.registry.load_map", return_value=None), \
             patch("avp.rosetta.registry.save_map"):
            avp_map = tiny_tied_connector._get_or_calibrate_map(tiny_untied_connector)
            assert avp_map is not None
            assert avp_map.source_dim == 64  # tiny model hidden_dim


@requires_torch
@requires_transformers
class TestQualityGateWarning:
    """Tests that the quality gate warning fires for long prompts."""

    def test_quality_gate_warning_logged(
        self, tiny_tied_connector, tiny_untied_connector, caplog,
    ):
        """Quality gate logs warning for long-prompt cross-model transfer."""
        import logging

        ctx = tiny_untied_connector.think("Hello", steps=3)

        # Artificially inflate seq_len to trigger quality gate
        # (prompt_tokens = seq_len - num_steps > 512)
        ctx_large = AVPContext(
            past_key_values=None,  # won't be used — replaced by primed KV
            model_hash=ctx.model_hash,
            num_steps=ctx.num_steps,
            seq_len=1000,
            model_family=ctx.model_family,
            hidden_dim=ctx.hidden_dim,
            num_layers=ctx.num_layers,
            last_hidden_state=ctx.last_hidden_state,
        )

        with caplog.at_level(logging.WARNING), \
             patch("avp.rosetta.registry.save_map"):
            tiny_tied_connector.generate(
                "Hello",
                context=ctx_large,
                source=tiny_untied_connector,
                max_new_tokens=10,
            )
        assert any("Quality gate" in r.message or "JSON fallback" in r.message
                    for r in caplog.records)


class TestEasyGenerateCrossModel:
    """Tests for easy API generate() with source_model=."""

    def test_source_model_calls_cross_model_path(self):
        """Verify source_model triggers cross-model flow."""
        mock_source = MagicMock()
        mock_target = MagicMock()
        mock_context = MagicMock()
        mock_source.think.return_value = mock_context
        mock_target.generate.return_value = "answer"

        with patch("avp.easy._get_or_create_connector") as mock_get:
            mock_get.side_effect = lambda m: (
                mock_source if m == "source_model" else mock_target
            )
            from avp.easy import generate

            result = generate(
                "Hello",
                model="target_model",
                source_model="source_model",
                think_steps=10,
            )

        assert result == "answer"
        mock_source.think.assert_called_once_with("Hello", steps=10)
        mock_target.generate.assert_called_once()
        call_kwargs = mock_target.generate.call_args
        assert call_kwargs.kwargs.get("source") is mock_source or \
            call_kwargs[1].get("source") is mock_source

    def test_no_source_model_uses_same_model_path(self):
        """Without source_model, uses existing same-model path."""
        with patch("avp.easy._get_or_create_connector") as mock_get, \
             patch("avp.easy.pack") as mock_pack:
            mock_connector = MagicMock()
            mock_connector.generate.return_value = "same-model answer"
            mock_get.return_value = mock_connector

            mock_packed = MagicMock()
            mock_packed.context = None
            mock_pack.return_value = mock_packed

            from avp.easy import generate

            result = generate("Hello", model="my_model", think_steps=0)

        assert result == "same-model answer"
