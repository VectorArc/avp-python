"""Tests for logit-guided cross-model decoding."""

import pytest
import torch
from conftest import requires_torch, requires_transformers


# ---------------------------------------------------------------------------
# Unit tests for compute_cross_model_logit_bias
# ---------------------------------------------------------------------------

@requires_torch
class TestComputeLogitBias:
    """Test bias computation from source hidden states."""

    def _make_avp_map(self, method, src_indices=None, tgt_indices=None,
                      target_norm=None):
        """Create a minimal AVPMap-like object for testing."""
        from avp.rosetta.calibrate import AVPMap

        return AVPMap(
            source_model_id="source",
            source_hash="src_hash",
            source_dim=32,
            target_model_id="target",
            target_hash="tgt_hash",
            target_dim=32,
            w_map=torch.randn(32, 32),
            bias=None,
            target_norm=target_norm or torch.tensor(1.0),
            method=method,
            anchor_count=0,
            validation_score=0.0,
            src_indices=src_indices,
            tgt_indices=tgt_indices,
        )

    def test_vocab_overlap_bias_shape(self):
        """Bias tensor has correct shape matching target vocab."""
        from avp.rosetta.logit_guided import compute_cross_model_logit_bias
        from avp.types import ProjectionMethod

        hidden = torch.randn(1, 32)
        lm_head_w = torch.randn(100, 32)  # source vocab=100
        target_vocab = 120

        src_idx = torch.arange(50)  # 50 shared tokens
        tgt_idx = torch.arange(50)

        avp_map = self._make_avp_map(
            ProjectionMethod.VOCAB_OVERLAP,
            src_indices=src_idx,
            tgt_indices=tgt_idx,
        )

        bias = compute_cross_model_logit_bias(
            hidden, lm_head_w, avp_map, target_vocab,
        )

        assert bias.shape == (target_vocab,)

    def test_vocab_overlap_unmapped_tokens_zero(self):
        """Unmapped tokens should have zero bias."""
        from avp.rosetta.logit_guided import compute_cross_model_logit_bias
        from avp.types import ProjectionMethod

        hidden = torch.randn(1, 32)
        lm_head_w = torch.randn(100, 32)
        target_vocab = 120

        src_idx = torch.arange(50)
        tgt_idx = torch.arange(50)  # only tokens 0-49 mapped

        avp_map = self._make_avp_map(
            ProjectionMethod.VOCAB_OVERLAP,
            src_indices=src_idx,
            tgt_indices=tgt_idx,
        )

        bias = compute_cross_model_logit_bias(
            hidden, lm_head_w, avp_map, target_vocab,
        )

        # Tokens 50-119 should be zero (before zero-mean adjustment)
        # After zero-mean, mapped tokens are shifted but unmapped stay zero
        unmapped_mask = torch.ones(target_vocab, dtype=torch.bool)
        unmapped_mask[tgt_idx] = False
        assert (bias[unmapped_mask] == 0.0).all()

    def test_vocab_overlap_bias_is_zero_mean(self):
        """Mapped tokens should have zero-mean bias."""
        from avp.rosetta.logit_guided import compute_cross_model_logit_bias
        from avp.types import ProjectionMethod

        hidden = torch.randn(1, 32)
        lm_head_w = torch.randn(100, 32)
        target_vocab = 120

        src_idx = torch.arange(50)
        tgt_idx = torch.arange(50)

        avp_map = self._make_avp_map(
            ProjectionMethod.VOCAB_OVERLAP,
            src_indices=src_idx,
            tgt_indices=tgt_idx,
        )

        bias = compute_cross_model_logit_bias(
            hidden, lm_head_w, avp_map, target_vocab,
        )

        mapped_bias = bias[tgt_idx]
        assert abs(mapped_bias.mean().item()) < 1e-5

    def test_vocab_mediated_bias(self):
        """VOCAB_MEDIATED uses direct 1:1 mapping."""
        from avp.rosetta.logit_guided import compute_cross_model_logit_bias
        from avp.types import ProjectionMethod

        hidden = torch.randn(1, 32)
        lm_head_w = torch.randn(100, 32)
        target_vocab = 100  # same size

        avp_map = self._make_avp_map(ProjectionMethod.VOCAB_MEDIATED)

        bias = compute_cross_model_logit_bias(
            hidden, lm_head_w, avp_map, target_vocab,
        )

        assert bias.shape == (target_vocab,)
        # All tokens should be mapped (non-zero after zero-mean)
        assert (bias != 0.0).sum() > 0

    def test_ridge_returns_zero_bias(self):
        """RIDGE method has no token-level mapping — returns zero bias."""
        from avp.rosetta.logit_guided import compute_cross_model_logit_bias
        from avp.types import ProjectionMethod

        hidden = torch.randn(1, 32)
        lm_head_w = torch.randn(100, 32)
        target_vocab = 120

        avp_map = self._make_avp_map(ProjectionMethod.RIDGE)

        bias = compute_cross_model_logit_bias(
            hidden, lm_head_w, avp_map, target_vocab,
        )

        assert (bias == 0.0).all()

    def test_1d_hidden_state(self):
        """Should handle 1D hidden state (no batch dim)."""
        from avp.rosetta.logit_guided import compute_cross_model_logit_bias
        from avp.types import ProjectionMethod

        hidden = torch.randn(32)  # no batch dim
        lm_head_w = torch.randn(100, 32)
        target_vocab = 120

        src_idx = torch.arange(50)
        tgt_idx = torch.arange(50)

        avp_map = self._make_avp_map(
            ProjectionMethod.VOCAB_OVERLAP,
            src_indices=src_idx,
            tgt_indices=tgt_idx,
        )

        bias = compute_cross_model_logit_bias(
            hidden, lm_head_w, avp_map, target_vocab,
        )

        assert bias.shape == (target_vocab,)


# ---------------------------------------------------------------------------
# Unit tests for CrossModelLogitBias processor
# ---------------------------------------------------------------------------

@requires_torch
class TestCrossModelLogitBias:
    """Test the LogitsProcessor behavior."""

    def test_applies_bias(self):
        """Bias should modify scores."""
        from avp.rosetta.logit_guided import CrossModelLogitBias

        bias = torch.tensor([0.0, 1.0, -1.0, 0.5])
        processor = CrossModelLogitBias(bias, alpha=1.0, confidence_threshold=1.0)

        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

        result = processor(input_ids, scores)

        # With uniform scores and confidence_threshold=1.0 (always apply),
        # result should be scores + alpha * bias
        expected = scores + bias
        assert torch.allclose(result, expected, atol=1e-5)

    def test_confidence_gating_suppresses_bias(self):
        """When target is confident, bias should be suppressed."""
        from avp.rosetta.logit_guided import CrossModelLogitBias

        bias = torch.tensor([0.0, 10.0, -10.0, 5.0])
        processor = CrossModelLogitBias(bias, alpha=1.0, confidence_threshold=0.5)

        input_ids = torch.tensor([[1]])
        # Very confident scores — token 0 dominates
        scores = torch.tensor([[100.0, -100.0, -100.0, -100.0]])

        result = processor(input_ids, scores)

        # Max prob ≈ 1.0 >> threshold 0.5, so bias should be suppressed
        assert torch.allclose(result, scores, atol=1e-5)

    def test_confidence_gating_allows_bias_when_uncertain(self):
        """When target is uncertain, bias should be applied."""
        from avp.rosetta.logit_guided import CrossModelLogitBias

        bias = torch.tensor([0.0, 1.0, -1.0, 0.5])
        processor = CrossModelLogitBias(bias, alpha=1.0, confidence_threshold=0.9)

        input_ids = torch.tensor([[1]])
        # Uniform scores — max_prob = 0.25 < 0.9
        scores = torch.zeros(1, 4)

        result = processor(input_ids, scores)

        # Should apply bias since target is uncertain
        expected = scores + bias
        assert torch.allclose(result, expected, atol=1e-5)

    def test_alpha_scaling(self):
        """Alpha should scale the bias."""
        from avp.rosetta.logit_guided import CrossModelLogitBias

        bias = torch.tensor([1.0, 2.0, 3.0])
        processor = CrossModelLogitBias(bias, alpha=0.5, confidence_threshold=1.0)

        input_ids = torch.tensor([[1]])
        scores = torch.zeros(1, 3)

        result = processor(input_ids, scores)
        expected = torch.tensor([[0.5, 1.0, 1.5]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_confidence_gating(self):
        """Per-batch-element gating should work."""
        from avp.rosetta.logit_guided import CrossModelLogitBias

        bias = torch.tensor([0.0, 1.0, -1.0])
        processor = CrossModelLogitBias(bias, alpha=1.0, confidence_threshold=0.5)

        input_ids = torch.tensor([[1], [2]])
        # Batch elem 0: confident (token 0 dominates)
        # Batch elem 1: uncertain (uniform)
        scores = torch.tensor([
            [100.0, -100.0, -100.0],
            [0.0, 0.0, 0.0],
        ])

        result = processor(input_ids, scores)

        # Elem 0: bias suppressed
        assert torch.allclose(result[0], scores[0], atol=1e-5)
        # Elem 1: bias applied
        expected_1 = scores[1] + bias
        assert torch.allclose(result[1], expected_1, atol=1e-5)
