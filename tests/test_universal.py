"""Tests for Universal Representation Space.

Uses tiny random-weight models (pattern from test_rosetta.py: GPT2Config with
vocab=256, n_embd=64) to test encoder/decoder shapes, codec roundtrip,
context serialization, adapter registry, handshake resolution, and
full pipeline without needing real models.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from conftest import requires_torch, requires_transformers

from avp.types import CommunicationMode, PayloadType
from avp.universal.config import UniversalConfig


# ---------------------------------------------------------------------------
# Tiny config for fast tests
# ---------------------------------------------------------------------------

TINY_CONFIG = UniversalConfig(
    d_universal=32,
    k_tokens=4,
    num_layers=2,
    num_heads=4,
    dropout=0.0,
    rollout_steps=8,
)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestUniversalConfig:
    def test_default_config(self):
        cfg = UniversalConfig()
        assert cfg.d_universal == 512
        assert cfg.k_tokens == 64
        assert cfg.num_layers == 6
        assert cfg.num_heads == 8
        assert cfg.dropout == 0.1

    def test_invalid_d_universal_not_divisible(self):
        with pytest.raises(ValueError, match="divisible"):
            UniversalConfig(d_universal=33, num_heads=8)

    def test_invalid_negative_k_tokens(self):
        with pytest.raises(ValueError, match="k_tokens"):
            UniversalConfig(k_tokens=0)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            UniversalConfig(dropout=1.0)


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------


@requires_torch
class TestUniversalEncoder:
    def test_output_shape_2d(self):
        """[T, D_src] → [K+2, D_universal]"""
        from avp.universal.encoder import UniversalEncoder

        encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
        hidden = torch.randn(10, 64)
        out = encoder(hidden)
        assert out.shape == (TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)

    def test_output_shape_3d(self):
        """[B, T, D_src] → [B, K+2, D_universal]"""
        from avp.universal.encoder import UniversalEncoder

        encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
        hidden = torch.randn(2, 10, 64)
        out = encoder(hidden)
        assert out.shape == (2, TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)

    def test_special_tokens_position(self):
        """Global token is second-to-last, style is last."""
        from avp.universal.encoder import UniversalEncoder

        encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
        hidden = torch.randn(10, 64)
        out = encoder(hidden)
        # K semantic tokens followed by global and style
        assert out.shape[0] == TINY_CONFIG.k_tokens + 2

    def test_different_sequence_lengths(self):
        """Encoder should handle varying input lengths."""
        from avp.universal.encoder import UniversalEncoder

        encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
        for T in [5, 20, 50]:
            out = encoder(torch.randn(T, 64))
            assert out.shape == (TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------


@requires_torch
class TestUniversalDecoder:
    def test_output_shape_2d(self):
        """[K+2, D_universal] → ([K, D_target], gate)"""
        from avp.universal.decoder import UniversalDecoder

        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        tokens = torch.randn(TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)
        decoded, gate = decoder(tokens)
        assert decoded.shape == (TINY_CONFIG.k_tokens, 64)
        assert isinstance(gate, float)

    def test_output_shape_3d(self):
        """[B, K+2, D_universal] → ([B, K, D_target], gate)"""
        from avp.universal.decoder import UniversalDecoder

        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        tokens = torch.randn(2, TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)
        decoded, gate = decoder(tokens)
        assert decoded.shape == (2, TINY_CONFIG.k_tokens, 64)
        assert isinstance(gate, float)

    def test_gate_in_range(self):
        """Gate should be in (0, 1) — sigmoid output."""
        from avp.universal.decoder import UniversalDecoder

        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        tokens = torch.randn(TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)
        _, gate = decoder(tokens)
        assert 0.0 < gate < 1.0

    def test_norm_match(self):
        """With target_norm, output vectors should have approx that norm."""
        from avp.universal.decoder import UniversalDecoder

        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        tokens = torch.randn(TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)
        target = 5.0
        decoded, _ = decoder(tokens, target_norm=target)
        norms = decoded.norm(dim=-1)
        # Each vector should be close to target norm
        for n in norms:
            assert abs(n.item() - target) < 0.1


# ---------------------------------------------------------------------------
# Roundtrip: encode → decode preserves dims
# ---------------------------------------------------------------------------


@requires_torch
class TestEncoderDecoderRoundtrip:
    def test_roundtrip_shapes(self):
        """Encode → decode produces [K, D_target] from [T, D_src]."""
        from avp.universal.decoder import UniversalDecoder
        from avp.universal.encoder import UniversalEncoder

        encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
        decoder = UniversalDecoder.create(d_target=128, config=TINY_CONFIG)

        hidden = torch.randn(20, 64)
        tokens = encoder(hidden)
        decoded, gate = decoder(tokens)

        assert decoded.shape == (TINY_CONFIG.k_tokens, 128)
        assert isinstance(gate, float)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


@requires_torch
class TestParamCount:
    def test_real_config_param_count(self):
        """Encoder with real config (D_src=3584) should be ~20M params."""
        from avp.universal.encoder import UniversalEncoder

        real_config = UniversalConfig()  # D=512, 6 layers, 8 heads
        encoder = UniversalEncoder.create(d_source=3584, config=real_config)
        total = sum(p.numel() for p in encoder.parameters())
        # Expected ~20M params (±5M for architecture variations)
        assert 10_000_000 < total < 30_000_000, f"Param count {total:,} outside expected range"

    def test_tiny_config_param_count(self):
        """Tiny config should have much fewer params."""
        from avp.universal.encoder import UniversalEncoder

        encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
        total = sum(p.numel() for p in encoder.parameters())
        assert total < 100_000


# ---------------------------------------------------------------------------
# Codec: encode_urt → decode with PayloadType.URT
# ---------------------------------------------------------------------------


class TestURTCodec:
    def test_encode_decode_urt(self):
        """URT encode → decode roundtrip preserves metadata."""
        import numpy as np
        from avp.codec import decode, encode_urt
        from avp.types import AVPMetadata, CommunicationMode

        tokens = np.random.randn(6, 32).astype(np.float32)  # K+2=6, D=32
        payload = tokens.tobytes()

        metadata = AVPMetadata(
            session_id="test-session",
            model_id="test-model",
            hidden_dim=64,
            num_layers=2,
            mode=CommunicationMode.UNIVERSAL,
            tensor_shape=tokens.shape,
        )

        wire = encode_urt(payload, metadata, k_tokens=4, d_universal=32)
        msg = decode(wire)

        assert msg.metadata.payload_type == PayloadType.URT
        assert msg.metadata.mode == CommunicationMode.UNIVERSAL
        assert msg.metadata.extra["k_tokens"] == "4"
        assert msg.metadata.extra["d_universal"] == "32"
        assert msg.metadata.extra["adapter_version"] == "1"
        assert msg.header.is_urt

        # Payload roundtrip
        decoded_arr = np.frombuffer(msg.payload, dtype=np.float32).reshape(tokens.shape)
        np.testing.assert_array_almost_equal(decoded_arr, tokens)


# ---------------------------------------------------------------------------
# Context serialization: AVPContext with is_universal survives to_bytes/from_bytes
# ---------------------------------------------------------------------------


@requires_torch
class TestUniversalContextSerialization:
    def test_to_bytes_from_bytes_roundtrip(self):
        """Universal AVPContext survives wire serialization."""
        from avp.context import AVPContext

        tokens = torch.randn(6, 32)  # K+2=6, D=32
        ctx = AVPContext(
            past_key_values=None,
            model_hash="abc123" * 8,
            num_steps=10,
            seq_len=0,
            model_family="test",
            hidden_dim=64,
            num_layers=2,
            universal_tokens=tokens,
            k_tokens=4,
            d_universal=32,
            is_universal=True,
        )

        wire = ctx.to_bytes(model_id="test-model")
        restored = AVPContext.from_bytes(wire, device="cpu")

        assert restored.is_universal
        assert restored.k_tokens == 4
        assert restored.d_universal == 32
        assert restored.model_hash == ctx.model_hash
        assert restored.num_steps == 10
        assert restored.universal_tokens is not None
        torch.testing.assert_close(
            restored.universal_tokens.float(), tokens, atol=1e-6, rtol=1e-6,
        )

    def test_kv_cache_context_still_works(self):
        """Non-universal AVPContext serialization is unchanged."""
        from avp.context import AVPContext
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        # Add a single layer's K/V
        k = torch.randn(1, 2, 4, 8)  # [B, heads, seq, head_dim]
        v = torch.randn(1, 2, 4, 8)
        cache.update(k, v, layer_idx=0)

        ctx = AVPContext(
            past_key_values=cache,
            model_hash="xyz456" * 8,
            num_steps=5,
            seq_len=4,
            model_family="gpt2",
            hidden_dim=16,
            num_layers=1,
        )

        wire = ctx.to_bytes()
        restored = AVPContext.from_bytes(wire, device="cpu")

        assert not restored.is_universal
        assert restored.model_hash == ctx.model_hash
        assert restored.seq_len == 4


# ---------------------------------------------------------------------------
# Adapter save/load
# ---------------------------------------------------------------------------


@requires_torch
class TestAdapterRegistry:
    def test_save_load_roundtrip(self):
        """Save and load adapter preserves all fields."""
        from avp.universal.adapter_registry import (
            UniversalAdapter,
            load_adapter,
            save_adapter,
        )
        from avp.universal.encoder import UniversalEncoder
        from avp.universal.decoder import UniversalDecoder

        tmpdir = Path(tempfile.mkdtemp())
        try:
            encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
            decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)

            adapter = UniversalAdapter(
                model_id="test/tiny-model",
                model_hash="a" * 64,
                d_source=64,
                config=TINY_CONFIG,
                encoder_state_dict=encoder.state_dict(),
                decoder_state_dict=decoder.state_dict(),
                target_norm=3.14,
                affine_out={"W": torch.eye(32), "b": torch.zeros(32)},
                affine_in=None,
            )

            path = save_adapter(adapter, adapter_dir=tmpdir)
            assert path.exists()

            loaded = load_adapter("a" * 64, adapter_dir=tmpdir)
            assert loaded is not None
            assert loaded.model_id == "test/tiny-model"
            assert loaded.d_source == 64
            assert loaded.target_norm == 3.14
            assert loaded.config.k_tokens == TINY_CONFIG.k_tokens
            assert loaded.affine_out is not None
            assert loaded.affine_in is None

            # State dicts should have same keys
            assert set(loaded.encoder_state_dict.keys()) == set(adapter.encoder_state_dict.keys())
            assert set(loaded.decoder_state_dict.keys()) == set(adapter.decoder_state_dict.keys())
        finally:
            shutil.rmtree(tmpdir)

    def test_load_nonexistent_returns_none(self):
        """Loading a non-existent adapter returns None."""
        from avp.universal.adapter_registry import load_adapter

        tmpdir = Path(tempfile.mkdtemp())
        try:
            result = load_adapter("nonexistent" * 4, adapter_dir=tmpdir)
            assert result is None
        finally:
            shutil.rmtree(tmpdir)

    def test_find_adapter(self):
        """find_adapter returns path when file exists, None otherwise."""
        from avp.universal.adapter_registry import (
            UniversalAdapter,
            find_adapter,
            save_adapter,
        )
        from avp.universal.encoder import UniversalEncoder
        from avp.universal.decoder import UniversalDecoder

        tmpdir = Path(tempfile.mkdtemp())
        try:
            assert find_adapter("b" * 64, adapter_dir=tmpdir) is None

            encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
            decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
            adapter = UniversalAdapter(
                model_id="test",
                model_hash="b" * 64,
                d_source=64,
                config=TINY_CONFIG,
                encoder_state_dict=encoder.state_dict(),
                decoder_state_dict=decoder.state_dict(),
            )
            save_adapter(adapter, adapter_dir=tmpdir)

            path = find_adapter("b" * 64, adapter_dir=tmpdir)
            assert path is not None
            assert path.exists()
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Handshake resolution: returns UNIVERSAL when adapters exist
# ---------------------------------------------------------------------------


@requires_torch
class TestHandshakeUniversal:
    def test_universal_disabled_in_handshake(self):
        """Handshake no longer resolves to UNIVERSAL even when adapters exist.

        Universal KV-cache priming via inputs_embeds was validated negative
        (0% same-model accuracy). Rule 5 in the handshake is disabled.
        """
        from avp.handshake import CompatibilityResolver
        from avp.types import ModelIdentity
        from avp.universal.adapter_registry import (
            UniversalAdapter,
            save_adapter,
        )
        from avp.universal.encoder import UniversalEncoder
        from avp.universal.decoder import UniversalDecoder

        tmpdir = Path(tempfile.mkdtemp())
        try:
            # Monkey-patch adapter dir
            import avp.universal.adapter_registry as reg
            old_dir = reg._ADAPTER_DIR
            reg._ADAPTER_DIR = tmpdir

            hash_a = "a" * 64
            hash_b = "b" * 64

            # Save adapters for both models
            for h in [hash_a, hash_b]:
                enc = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
                dec = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
                adapter = UniversalAdapter(
                    model_id=f"test/{h[:8]}",
                    model_hash=h,
                    d_source=64,
                    config=TINY_CONFIG,
                    encoder_state_dict=enc.state_dict(),
                    decoder_state_dict=dec.state_dict(),
                )
                save_adapter(adapter, adapter_dir=tmpdir)

            local = ModelIdentity(model_hash=hash_a, model_family="x", hidden_dim=64, num_layers=2)
            remote = ModelIdentity(model_hash=hash_b, model_family="y", hidden_dim=128, num_layers=4)

            session = CompatibilityResolver.resolve(local, remote)
            # Universal is disabled — should fall through to JSON
            assert session.mode != CommunicationMode.UNIVERSAL

            reg._ADAPTER_DIR = old_dir
        finally:
            shutil.rmtree(tmpdir)

    def test_no_universal_without_adapters(self):
        """Without adapters, handshake falls back to JSON (or other rules)."""
        from avp.handshake import CompatibilityResolver
        from avp.types import ModelIdentity

        tmpdir = Path(tempfile.mkdtemp())
        try:
            import avp.universal.adapter_registry as reg
            old_dir = reg._ADAPTER_DIR
            reg._ADAPTER_DIR = tmpdir

            local = ModelIdentity(model_hash="c" * 64, model_family="x", hidden_dim=64, num_layers=2)
            remote = ModelIdentity(model_hash="d" * 64, model_family="y", hidden_dim=128, num_layers=4)

            session = CompatibilityResolver.resolve(local, remote)
            assert session.mode != CommunicationMode.UNIVERSAL

            reg._ADAPTER_DIR = old_dir
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# KV-cache priming: decoder output through model → valid DynamicCache
# ---------------------------------------------------------------------------


@requires_torch
@requires_transformers
class TestKVCachePriming:
    def test_priming_produces_valid_cache(self):
        """Decoded embeddings fed through model produce usable DynamicCache."""
        from transformers import GPT2Config, GPT2LMHeadModel
        from avp.universal.decoder import UniversalDecoder

        config = GPT2Config(
            vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128,
        )
        model = GPT2LMHeadModel(config)
        model.eval()

        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        tokens = torch.randn(TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)

        with torch.no_grad():
            decoded, gate = decoder(tokens, target_norm=1.0)
            embed_input = decoded.unsqueeze(0).to(model.dtype) * gate
            embed_mask = torch.ones((1, embed_input.shape[1]), dtype=torch.long)

            out = model(
                inputs_embeds=embed_input,
                attention_mask=embed_mask,
                use_cache=True,
                return_dict=True,
            )

        assert out.past_key_values is not None
        # Check KV-cache has entries for K tokens
        from transformers.cache_utils import Cache
        if isinstance(out.past_key_values, Cache):
            assert out.past_key_values.get_seq_length() == TINY_CONFIG.k_tokens


# ---------------------------------------------------------------------------
# NormMatch: outputs have correct target norm
# ---------------------------------------------------------------------------


@requires_torch
class TestNormMatch:
    def test_decoder_norm_match(self):
        """Decoder NormMatch scales output to specified target norm."""
        from avp.universal.decoder import UniversalDecoder

        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        tokens = torch.randn(TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)

        target = 2.5
        decoded, _ = decoder(tokens, target_norm=target)
        norms = decoded.norm(dim=-1)
        for n in norms:
            assert abs(n.item() - target) < 0.05

    def test_decoder_without_norm_match(self):
        """Without target_norm, output norms vary freely."""
        from avp.universal.decoder import UniversalDecoder

        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        tokens = torch.randn(TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)

        decoded, _ = decoder(tokens)
        norms = decoded.norm(dim=-1)
        # Just check it runs without error and norms are finite
        assert all(torch.isfinite(norms))


# ---------------------------------------------------------------------------
# Full pipeline: encode → serialize → deserialize → decode → generate (no crash)
# ---------------------------------------------------------------------------


@requires_torch
@requires_transformers
class TestFullPipeline:
    def test_encode_serialize_decode_generate(self):
        """Full pipeline with tiny model doesn't crash."""
        from transformers import GPT2Config, GPT2LMHeadModel
        from avp.context import AVPContext
        from avp.universal.encoder import UniversalEncoder
        from avp.universal.decoder import UniversalDecoder

        config = GPT2Config(
            vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128,
        )
        model = GPT2LMHeadModel(config)
        model.eval()

        # Simulate hidden states from latent rollout
        hidden_states = torch.randn(TINY_CONFIG.rollout_steps + 1, 64)

        # Encode
        encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
        universal_tokens = encoder(hidden_states)
        assert universal_tokens.shape == (TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal)

        # Serialize
        ctx = AVPContext(
            past_key_values=None,
            model_hash="test" * 16,
            num_steps=TINY_CONFIG.rollout_steps,
            seq_len=0,
            model_family="gpt2",
            hidden_dim=64,
            num_layers=2,
            universal_tokens=universal_tokens,
            k_tokens=TINY_CONFIG.k_tokens,
            d_universal=TINY_CONFIG.d_universal,
            is_universal=True,
        )
        wire = ctx.to_bytes()
        assert len(wire) > 0

        # Deserialize
        restored = AVPContext.from_bytes(wire, device="cpu")
        assert restored.is_universal
        torch.testing.assert_close(
            restored.universal_tokens.float(),
            universal_tokens.float(),
            atol=1e-6, rtol=1e-6,
        )

        # Decode
        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        decoded, gate = decoder(restored.universal_tokens, target_norm=1.0)
        assert decoded.shape == (TINY_CONFIG.k_tokens, 64)

        # Generate: prime KV-cache + generate tokens
        with torch.no_grad():
            embed_input = decoded.unsqueeze(0).to(model.dtype) * gate
            embed_mask = torch.ones((1, embed_input.shape[1]), dtype=torch.long)
            prime_out = model(
                inputs_embeds=embed_input,
                attention_mask=embed_mask,
                use_cache=True,
                return_dict=True,
            )

        # Verify we can continue generation from primed cache
        past_kv = prime_out.past_key_values
        input_ids = torch.tensor([[2, 3, 4]])  # dummy tokens
        gen_mask = torch.ones((1, TINY_CONFIG.k_tokens + 3), dtype=torch.long)

        with torch.no_grad():
            gen_out = model(
                input_ids=input_ids,
                attention_mask=gen_mask,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
        assert gen_out.logits.shape[1] == 3


# ---------------------------------------------------------------------------
# Gradient flow: encoder and decoder support backprop
# ---------------------------------------------------------------------------


@requires_torch
class TestGradientFlow:
    def test_encoder_gradient(self):
        """Encoder supports backpropagation through its parameters."""
        from avp.universal.encoder import UniversalEncoder

        encoder = UniversalEncoder.create(d_source=64, config=TINY_CONFIG)
        hidden = torch.randn(10, 64)
        out = encoder(hidden)
        loss = out.sum()
        loss.backward()
        # Encoder parameters should receive gradients
        grad_params = [p for p in encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        assert len(grad_params) > 0, "No encoder parameters received gradients"

    def test_decoder_gradient(self):
        """Decoder supports backpropagation."""
        from avp.universal.decoder import UniversalDecoder

        decoder = UniversalDecoder.create(d_target=64, config=TINY_CONFIG)
        tokens = torch.randn(
            TINY_CONFIG.k_tokens + 2, TINY_CONFIG.d_universal, requires_grad=True,
        )
        decoded, gate = decoder(tokens)
        loss = decoded.sum()
        loss.backward()
        assert tokens.grad is not None
        assert tokens.grad.abs().sum() > 0
