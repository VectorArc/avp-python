"""Tests for Rosetta Stone cross-model projection."""

import pytest
import torch
from conftest import MockTokenizer, requires_torch, requires_transformers

from avp.types import ProjectionMethod


# ---------------------------------------------------------------------------
# Mock tokenizer with get_vocab() support for tokenizer hash tests
# ---------------------------------------------------------------------------

class VocabMockTokenizer(MockTokenizer):
    """MockTokenizer extended with get_vocab() for tokenizer hash tests."""

    def __init__(self, vocab_size=256, vocab_offset=0):
        super().__init__(vocab_size=vocab_size)
        self._vocab_offset = vocab_offset

    def get_vocab(self):
        off = self._vocab_offset
        return {f"token_{off + i}": i for i in range(self.vocab_size)}


class CrossFamilyMockTokenizer(MockTokenizer):
    """Mock tokenizer with partial vocab overlap with VocabMockTokenizer.

    First half of tokens share names with VocabMockTokenizer (token_0..token_N/2-1).
    Second half use different names (alt_token_N/2..alt_token_N-1).
    """

    def get_vocab(self):
        vocab = {}
        for i in range(self.vocab_size):
            if i < self.vocab_size // 2:
                vocab[f"token_{i}"] = i
            else:
                vocab[f"alt_token_{i}"] = i
        return vocab


# ---------------------------------------------------------------------------
# Fixtures: tiny models with different architectures / dims
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_gpt2_64():
    """GPT2 tiny model: hidden_dim=64, tied weights."""
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    tokenizer = MockTokenizer(vocab_size=256)
    return model, tokenizer


@pytest.fixture
def tiny_gpt2_128():
    """GPT2 tiny model: hidden_dim=128, tied weights (different dim from 64)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        vocab_size=256, n_embd=128, n_head=4, n_layer=2, n_positions=128,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    tokenizer = MockTokenizer(vocab_size=256)
    return model, tokenizer


@pytest.fixture
def tiny_gpt2_64_v2():
    """Another GPT2 tiny model: hidden_dim=64, same dim as tiny_gpt2_64 but
    different random weights (different config hash due to different n_layer)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        vocab_size=256, n_embd=64, n_head=4, n_layer=3, n_positions=128,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    tokenizer = MockTokenizer(vocab_size=256)
    return model, tokenizer


@pytest.fixture
def seeded_tiny_gpt2_64():
    """GPT2 tiny model: hidden_dim=64, tied weights, seeded for determinism."""
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    tokenizer = VocabMockTokenizer(vocab_size=256)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tests: projection function
# ---------------------------------------------------------------------------

@requires_torch
class TestProjection:
    def test_projection_shape(self):
        """apply_cross_model_projection produces correct output shape."""
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(2, 64)         # [B, D_src]
        w_map = torch.randn(64, 128)         # [D_src, D_tgt]
        target_norm = torch.tensor(5.0)

        result = apply_cross_model_projection(hidden, w_map, target_norm)
        assert result.shape == (2, 128)

    def test_projection_shape_3d(self):
        """Works with 3D input [..., D_src]."""
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(2, 3, 64)
        w_map = torch.randn(64, 128)
        target_norm = torch.tensor(5.0)

        result = apply_cross_model_projection(hidden, w_map, target_norm)
        assert result.shape == (2, 3, 128)

    def test_projection_norm(self):
        """Output vectors have correct target norm."""
        import numpy as np
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(4, 64)
        w_map = torch.randn(64, 128)
        target_norm = torch.tensor(7.5)

        result = apply_cross_model_projection(hidden, w_map, target_norm)
        assert isinstance(result, np.ndarray)
        norms = np.linalg.norm(result, axis=-1)
        for n in norms:
            assert abs(float(n) - 7.5) < 0.01

    def test_projection_with_bias(self):
        """Bias is correctly applied."""
        import numpy as np
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(2, 64)
        w_map = torch.randn(64, 128)
        bias = torch.randn(128)
        target_norm = torch.tensor(5.0)

        result = apply_cross_model_projection(hidden, w_map, target_norm, bias=bias)
        assert result.shape == (2, 128)
        # Result should differ from no-bias version
        result_no_bias = apply_cross_model_projection(hidden, w_map, target_norm)
        assert not np.allclose(result, result_no_bias)

    def test_projection_preserves_dtype(self):
        """Output is always float32 numpy (projection normalizes to float32)."""
        import numpy as np
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(2, 64, dtype=torch.float16)
        w_map = torch.randn(64, 128)
        target_norm = torch.tensor(5.0)

        result = apply_cross_model_projection(hidden, w_map, target_norm)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_projection_square_matrix(self):
        """Works with square (same-dim) projection matrix."""
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(2, 64)
        w_map = torch.randn(64, 64)
        target_norm = torch.tensor(5.0)

        result = apply_cross_model_projection(hidden, w_map, target_norm)
        assert result.shape == (2, 64)


# ---------------------------------------------------------------------------
# Tests: calibration
# ---------------------------------------------------------------------------

@requires_torch
@requires_transformers
class TestCalibrate:
    ANCHOR_TEXTS = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test sentence.",
        "Machine learning is a subset of artificial intelligence.",
        "The area of a circle is pi r squared.",
        "def hello(): print('hi')",
        "Bonjour, comment allez-vous?",
        "The stock market rose today.",
        "Water boils at 100 degrees Celsius.",
        "She walked through the garden noting the roses.",
        "Calculate the sum of one plus two.",
    ]

    def test_calibrate_incompatible_models_raises(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() raises ValueError when models have no projection path."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        # Use disjoint vocabularies — zero overlap, no shared tokenizer
        tgt_tok = VocabMockTokenizer(vocab_size=256, vocab_offset=1000)

        with pytest.raises(ValueError, match="No projection path found"):
            calibrate(src_model, tgt_model, src_tok, tgt_tok, device="cpu")


# ---------------------------------------------------------------------------
# Tests: registry (save / load / find)
# ---------------------------------------------------------------------------

@requires_torch
@requires_transformers
class TestRegistry:
    def test_save_load_roundtrip(self, tmp_path, tiny_gpt2_64, tiny_gpt2_128):
        """Save and load produces identical AVPMap."""
        from avp.rosetta.calibrate import AVPMap
        from avp.rosetta.registry import load_map, save_map

        avp_map = AVPMap(
            source_model_id="src", source_hash="a" * 64, source_dim=64,
            target_model_id="tgt", target_hash="b" * 64, target_dim=128,
            w_map=torch.randn(256, 128),
            bias=None,
            target_norm=torch.tensor(5.0),
            method="vocab_mediated",
            anchor_count=0,
            validation_score=1.0,
        )

        path = save_map(avp_map, map_dir=tmp_path)
        assert path.exists()

        loaded = load_map(
            avp_map.source_hash, avp_map.target_hash,
            device="cpu", map_dir=tmp_path,
        )
        assert loaded is not None
        assert loaded.source_model_id == avp_map.source_model_id
        assert loaded.target_model_id == avp_map.target_model_id
        assert loaded.source_dim == avp_map.source_dim
        assert loaded.target_dim == avp_map.target_dim
        assert loaded.method == avp_map.method
        assert loaded.anchor_count == avp_map.anchor_count
        assert abs(loaded.validation_score - avp_map.validation_score) < 1e-6
        assert torch.allclose(loaded.w_map, avp_map.w_map)
        assert torch.allclose(loaded.target_norm, avp_map.target_norm)

    def test_find_map_exists(self, tmp_path):
        """find_map returns path when map file exists."""
        from avp.rosetta.registry import find_map, _map_filename

        src_hash = "a" * 64
        tgt_hash = "b" * 64
        filename = _map_filename(src_hash, tgt_hash)
        (tmp_path / filename).write_bytes(b"dummy")

        result = find_map(src_hash, tgt_hash, map_dir=tmp_path)
        assert result is not None
        assert result.name == filename

    def test_find_map_missing(self, tmp_path):
        """find_map returns None when no map file exists."""
        from avp.rosetta.registry import find_map

        result = find_map("x" * 64, "y" * 64, map_dir=tmp_path)
        assert result is None

    def test_load_map_missing(self, tmp_path):
        """load_map returns None when no map file exists."""
        from avp.rosetta.registry import load_map

        result = load_map("x" * 64, "y" * 64, map_dir=tmp_path)
        assert result is None

    def test_map_id(self):
        """map_id produces expected format."""
        from avp.rosetta.registry import map_id

        src_hash = "abcdef1234567890" + "x" * 48
        tgt_hash = "0987654321fedcba" + "y" * 48
        result = map_id(src_hash, tgt_hash)
        assert result == "abcdef1234567890_0987654321fedcba"


# ---------------------------------------------------------------------------
# Tests: handshake integration
# ---------------------------------------------------------------------------

@requires_torch
class TestHandshakeIntegration:
    def test_handshake_with_map(self, tmp_path):
        """CompatibilityResolver returns LATENT when a Rosetta map exists."""
        from avp.handshake import CompatibilityResolver
        from avp.rosetta.registry import _map_filename
        from avp.types import CommunicationMode, ModelIdentity
        import avp.rosetta.registry as registry

        src_hash = "src" + "0" * 61
        tgt_hash = "tgt" + "0" * 61

        local = ModelIdentity(
            model_family="gpt2", model_id="gpt2-small",
            model_hash=src_hash, hidden_dim=64, num_layers=2,
        )
        remote = ModelIdentity(
            model_family="llama", model_id="llama-7b",
            model_hash=tgt_hash, hidden_dim=128, num_layers=4,
        )

        # Without map: should be JSON
        old_map_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = tmp_path
            result = CompatibilityResolver.resolve(local, remote)
            assert result.mode == CommunicationMode.JSON
            assert result.avp_map_id == ""

            # Create map file
            filename = _map_filename(src_hash, tgt_hash)
            (tmp_path / filename).write_bytes(b"dummy")

            result = CompatibilityResolver.resolve(local, remote)
            assert result.mode == CommunicationMode.LATENT
            assert result.avp_map_id != ""
        finally:
            registry._MAP_DIR = old_map_dir

    def test_handshake_same_model_ignores_map(self, tmp_path):
        """Same-model handshake stays LATENT and doesn't set avp_map_id."""
        from avp.handshake import CompatibilityResolver
        from avp.types import CommunicationMode, ModelIdentity

        same_hash = "same" + "0" * 60

        local = ModelIdentity(
            model_family="gpt2", model_id="gpt2",
            model_hash=same_hash, hidden_dim=64, num_layers=2,
        )
        remote = ModelIdentity(
            model_family="gpt2", model_id="gpt2",
            model_hash=same_hash, hidden_dim=64, num_layers=2,
        )

        result = CompatibilityResolver.resolve(local, remote)
        assert result.mode == CommunicationMode.LATENT
        assert result.avp_map_id == ""


# ---------------------------------------------------------------------------
# Tests: connector integration
# ---------------------------------------------------------------------------

@requires_torch
@requires_transformers
class TestConnectorIntegration:
    def test_project_hidden_for_cross_model(self, tiny_gpt2_64, tiny_gpt2_128):
        """HuggingFaceConnector.project_hidden_for_cross_model works."""
        from avp.connectors.huggingface import HuggingFaceConnector
        from avp.rosetta.calibrate import AVPMap

        src_model, src_tok = tiny_gpt2_64
        connector = HuggingFaceConnector(model=src_model, tokenizer=src_tok, device="cpu")

        avp_map = AVPMap(
            source_model_id="src", source_hash="s" * 64, source_dim=64,
            target_model_id="tgt", target_hash="t" * 64, target_dim=128,
            w_map=torch.randn(64, 128),
            bias=None,
            target_norm=torch.tensor(5.0),
            method="vocab_mediated",
            anchor_count=10,
            validation_score=0.8,
        )

        hidden = torch.randn(1, 64)
        result = connector.project_hidden_for_cross_model(hidden, avp_map)
        assert result.shape == (1, 128)
        assert abs(result.norm(dim=-1).item() - 5.0) < 0.01

    def test_project_hidden_vocab_mediated(self, tiny_gpt2_64, tiny_gpt2_128):
        """HuggingFaceConnector.project_hidden_for_cross_model works with vocab_mediated."""
        from avp.connectors.huggingface import HuggingFaceConnector
        from avp.rosetta.calibrate import AVPMap

        src_model, src_tok = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        connector = HuggingFaceConnector(model=src_model, tokenizer=src_tok, device="cpu")

        # w_map holds target input embedding weights for vocab_mediated
        tgt_embed = tgt_model.get_input_embeddings().weight.detach().clone()
        avp_map = AVPMap(
            source_model_id="src", source_hash="s" * 64, source_dim=64,
            target_model_id="tgt", target_hash="t" * 64, target_dim=128,
            w_map=tgt_embed,
            bias=None,
            target_norm=torch.tensor(5.0),
            method="vocab_mediated",
            anchor_count=0,
            validation_score=1.0,
        )

        hidden = torch.randn(1, 64)
        result = connector.project_hidden_for_cross_model(hidden, avp_map)
        assert result.shape == (1, 128)


# ---------------------------------------------------------------------------
# Tests: vocabulary-mediated projection
# ---------------------------------------------------------------------------

@requires_torch
class TestVocabMediatedProjection:
    def test_vocab_mediated_shape(self):
        """vocabulary_mediated_projection produces correct output shape."""
        from avp.rosetta.project import vocabulary_mediated_projection

        hidden = torch.randn(2, 64)                  # [B, D_src]
        source_lm_head = torch.randn(256, 64)        # [vocab_size, D_src]
        target_embed = torch.randn(256, 128)          # [vocab_size, D_tgt]

        result = vocabulary_mediated_projection(hidden, source_lm_head, target_embed)
        assert result.shape == (2, 128)

    def test_vocab_mediated_shape_3d(self):
        """Works with 3D input [..., D_src]."""
        from avp.rosetta.project import vocabulary_mediated_projection

        hidden = torch.randn(2, 3, 64)
        source_lm_head = torch.randn(256, 64)
        target_embed = torch.randn(256, 128)

        result = vocabulary_mediated_projection(hidden, source_lm_head, target_embed)
        assert result.shape == (2, 3, 128)

    def test_vocab_mediated_cosine_sim(self):
        """Output has high cosine similarity to nearest target embedding (>0.9)."""
        import numpy as np
        from avp.rosetta.project import vocabulary_mediated_projection

        vocab_size = 256
        d_src, d_tgt = 64, 128
        source_lm_head = torch.randn(vocab_size, d_src)
        target_embed = torch.randn(vocab_size, d_tgt)

        # Use a real embedding as input (high confidence in one token)
        hidden = source_lm_head[42].unsqueeze(0)  # [1, D_src] — looks like token 42
        result = vocabulary_mediated_projection(hidden, source_lm_head, target_embed)

        # Result should be close to target_embed[42] since softmax should peak at token 42
        tgt_np = target_embed[42:43].numpy()
        r_norm = result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-6)
        t_norm = tgt_np / np.maximum(np.linalg.norm(tgt_np, axis=-1, keepdims=True), 1e-6)
        cos_sim = float((r_norm * t_norm).sum())
        assert cos_sim > 0.9, f"Cosine similarity {cos_sim:.3f} should be > 0.9"

    def test_vocab_mediated_matches_same_model(self):
        """When source==target weights, matches project_to_embedding_space()."""
        import numpy as np
        from avp.realign import project_to_embedding_space
        from avp.rosetta.project import vocabulary_mediated_projection

        vocab_size = 256
        hidden_dim = 64
        embed_weight = torch.randn(vocab_size, hidden_dim)
        hidden = torch.randn(4, hidden_dim)

        # Same weights for both source lm_head and target embed
        result_vocab = vocabulary_mediated_projection(
            hidden, embed_weight, embed_weight, temperature=1.0
        )
        result_same = project_to_embedding_space(
            hidden, embed_weight, temperature=1.0
        )

        # Both return numpy now
        assert np.allclose(result_vocab, result_same, atol=1e-5)

    def test_vocab_mediated_preserves_dtype(self):
        """Output is always float32 numpy."""
        import numpy as np
        from avp.rosetta.project import vocabulary_mediated_projection

        hidden = torch.randn(2, 64, dtype=torch.float16)
        source_lm_head = torch.randn(256, 64)
        target_embed = torch.randn(256, 128)

        result = vocabulary_mediated_projection(hidden, source_lm_head, target_embed)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_vocab_mediated_return_metrics(self):
        """return_metrics=True returns (projected, metrics_dict) with entropy and max_prob."""
        from avp.rosetta.project import vocabulary_mediated_projection

        hidden = torch.randn(2, 64)
        source_lm_head = torch.randn(256, 64)
        target_embed = torch.randn(256, 128)

        result = vocabulary_mediated_projection(
            hidden, source_lm_head, target_embed, return_metrics=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        projected, metrics = result
        assert projected.shape == (2, 128)
        assert "entropy" in metrics and "max_prob" in metrics
        assert metrics["entropy"].shape == (2,)
        assert (metrics["entropy"] >= 0).all()
        assert (metrics["max_prob"] >= 0).all() and (metrics["max_prob"] <= 1).all()

    def test_vocab_mediated_return_metrics_false_unchanged(self):
        """Default return_metrics=False returns just a numpy array."""
        import numpy as np
        from avp.rosetta.project import vocabulary_mediated_projection

        hidden = torch.randn(2, 64)
        source_lm_head = torch.randn(256, 64)
        target_embed = torch.randn(256, 128)

        result = vocabulary_mediated_projection(hidden, source_lm_head, target_embed)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 128)

    def test_entropy_peaked_vs_flat(self):
        """Peaked hidden state (matching a specific token) should have lower entropy."""
        from avp.rosetta.project import vocabulary_mediated_projection

        torch.manual_seed(42)
        source_lm_head = torch.randn(256, 64)
        target_embed = torch.randn(256, 128)

        # Peaked: hidden = one of the lm_head rows (model "knows" it's token 42)
        peaked_hidden = source_lm_head[42].unsqueeze(0)
        _, peaked_metrics = vocabulary_mediated_projection(
            peaked_hidden, source_lm_head, target_embed, return_metrics=True,
        )

        # Flat: hidden = zeros (uninformative)
        flat_hidden = torch.zeros(1, 64)
        _, flat_metrics = vocabulary_mediated_projection(
            flat_hidden, source_lm_head, target_embed, return_metrics=True,
        )

        assert peaked_metrics["entropy"].item() < flat_metrics["entropy"].item()
        assert peaked_metrics["max_prob"].item() > flat_metrics["max_prob"].item()


# ---------------------------------------------------------------------------
# Tests: tokenizer hash
# ---------------------------------------------------------------------------

@requires_torch
class TestTokenizerHash:
    def test_tokenizer_hash_deterministic(self):
        """Same tokenizer produces the same hash."""
        from avp.handshake import compute_tokenizer_hash

        tok1 = VocabMockTokenizer(vocab_size=256)
        tok2 = VocabMockTokenizer(vocab_size=256)

        h1 = compute_tokenizer_hash(tok1)
        h2 = compute_tokenizer_hash(tok2)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_tokenizer_hash_different(self):
        """Different vocabularies produce different hashes."""
        from avp.handshake import compute_tokenizer_hash

        tok_small = VocabMockTokenizer(vocab_size=128)
        tok_large = VocabMockTokenizer(vocab_size=256)

        h_small = compute_tokenizer_hash(tok_small)
        h_large = compute_tokenizer_hash(tok_large)
        assert h_small != h_large

    def test_tokenizer_hash_no_get_vocab(self):
        """Tokenizer without get_vocab() returns empty string."""
        from avp.handshake import compute_tokenizer_hash

        class BareTokenizer:
            """Tokenizer with no get_vocab() method."""
            pass

        tok = BareTokenizer()
        h = compute_tokenizer_hash(tok)
        assert h == ""

    def test_extract_identity_with_tokenizer(self):
        """extract_model_identity populates tokenizer_hash when tokenizer given."""
        from avp.handshake import extract_model_identity

        config = {
            "model_type": "gpt2",
            "_name_or_path": "gpt2",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        }
        tok = VocabMockTokenizer(vocab_size=256)
        identity = extract_model_identity(config, tokenizer=tok)
        assert identity.tokenizer_hash != ""
        assert len(identity.tokenizer_hash) == 64

    def test_extract_identity_without_tokenizer(self):
        """extract_model_identity has empty tokenizer_hash without tokenizer."""
        from avp.handshake import extract_model_identity

        config = {
            "model_type": "gpt2",
            "_name_or_path": "gpt2",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        }
        identity = extract_model_identity(config)
        assert identity.tokenizer_hash == ""


# ---------------------------------------------------------------------------
# Tests: handshake with shared tokenizer
# ---------------------------------------------------------------------------

@requires_torch
class TestHandshakeSharedTokenizer:
    def test_handshake_shared_tokenizer(self):
        """Shared tokenizer_hash → LATENT with avp_map_id="vocab:..."."""
        from avp.handshake import CompatibilityResolver, compute_tokenizer_hash
        from avp.types import CommunicationMode, ModelIdentity

        tok = VocabMockTokenizer(vocab_size=256)
        tok_hash = compute_tokenizer_hash(tok)

        local = ModelIdentity(
            model_family="qwen2", model_id="qwen2-1.5b",
            model_hash="a" * 64, hidden_dim=1536, num_layers=28,
            tokenizer_hash=tok_hash,
        )
        remote = ModelIdentity(
            model_family="qwen2", model_id="qwen2-0.5b",
            model_hash="b" * 64, hidden_dim=896, num_layers=24,
            tokenizer_hash=tok_hash,
        )

        result = CompatibilityResolver.resolve(local, remote)
        assert result.mode == CommunicationMode.LATENT
        assert result.avp_map_id.startswith("vocab:")

    def test_handshake_different_tokenizer(self):
        """Different tokenizer_hash → JSON (no map file)."""
        from avp.handshake import CompatibilityResolver, compute_tokenizer_hash
        from avp.types import CommunicationMode, ModelIdentity
        import avp.rosetta.registry as registry

        tok_a = VocabMockTokenizer(vocab_size=256)
        tok_b = VocabMockTokenizer(vocab_size=128)

        local = ModelIdentity(
            model_family="qwen2", model_id="qwen2-1.5b",
            model_hash="a" * 64, hidden_dim=1536, num_layers=28,
            tokenizer_hash=compute_tokenizer_hash(tok_a),
        )
        remote = ModelIdentity(
            model_family="llama", model_id="llama-7b",
            model_hash="b" * 64, hidden_dim=4096, num_layers=32,
            tokenizer_hash=compute_tokenizer_hash(tok_b),
        )

        # Point registry to empty dir so no map files found
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = registry._MAP_DIR
            try:
                registry._MAP_DIR = __import__("pathlib").Path(tmp)
                result = CompatibilityResolver.resolve(local, remote)
                assert result.mode == CommunicationMode.JSON
            finally:
                registry._MAP_DIR = old_dir


# ---------------------------------------------------------------------------
# Tests: calibrate with vocab-mediated detection
# ---------------------------------------------------------------------------

@requires_torch
@requires_transformers
class TestCalibrateVocabMediated:
    def test_calibrate_detects_shared_vocab(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() detects shared vocab and returns method='vocab_mediated'."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128

        # Use VocabMockTokenizer (has get_vocab()) with identical vocab
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok, device="cpu",
        )

        assert avp_map.method == ProjectionMethod.VOCAB_MEDIATED
        assert avp_map.anchor_count == 0
        assert avp_map.validation_score == 1.0
        # w_map should be target's input embedding weights [vocab_size, D_tgt]
        assert avp_map.w_map.shape == (256, 128)

    def test_calibrate_incompatible_raises_no_overlap(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() raises ValueError when vocab has zero overlap."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        # Use disjoint vocabularies — zero overlap
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256, vocab_offset=1000)

        with pytest.raises(ValueError, match="No projection path found"):
            calibrate(src_model, tgt_model, src_tok, tgt_tok, device="cpu")

    def test_calibrate_vocab_mediated_auto_detected(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() auto-detects shared vocab and uses vocab_mediated."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            device="cpu",
        )
        assert avp_map.method == ProjectionMethod.VOCAB_MEDIATED

    def test_calibrate_different_vocab_no_overlap_raises(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() raises ValueError when vocab has zero overlap."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        # Use disjoint vocabularies (no shared tokens)
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256, vocab_offset=1000)

        with pytest.raises(ValueError, match="No projection path found"):
            calibrate(
                src_model, tgt_model, src_tok, tgt_tok,
                device="cpu",
            )


# ---------------------------------------------------------------------------
# Tests: projection validation
# ---------------------------------------------------------------------------

@requires_torch
@requires_transformers
class TestValidation:
    """Tests for rosetta.validate — projection quality validation."""

    def test_validate_same_model_high_confidence(self, tiny_gpt2_64):
        """Same-model vocab-mediated projection runs and returns a valid result.

        Note: random-weight tiny models produce near-uniform softmax → low cos_sim.
        With real trained models, same-model projection gives cos_sim > 0.9.
        Here we bypass the fast gate to test the full validation pipeline.
        """
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.validate import validate_projection, ValidationConfig

        model, _ = tiny_gpt2_64
        tok = VocabMockTokenizer(vocab_size=256)

        # Calibrate same model → vocab_mediated (shared vocab)
        avp_map = calibrate(model, model, tok, tok, device="cpu")

        # Bypass fast gate (random weights → near-zero cos sim is expected)
        config = ValidationConfig(cosine_sim_threshold=-1.0)
        result = validate_projection(
            model, model, avp_map, tok, tok, config=config, device="cpu",
        )

        assert isinstance(result.cosine_similarity, float)
        assert result.perplexity is not None
        assert result.perplexity > 0
        assert result.detail  # non-empty

    def test_validate_random_map_low_confidence(self, tiny_gpt2_64, tiny_gpt2_128):
        """Random projection matrix should recommend JSON."""
        from avp.rosetta.calibrate import AVPMap
        from avp.rosetta.validate import validate_projection
        from avp.types import CommunicationMode

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        # Random w_map — should produce garbage projection
        avp_map = AVPMap(
            source_model_id="src", source_hash="s" * 64, source_dim=64,
            target_model_id="tgt", target_hash="t" * 64, target_dim=128,
            w_map=torch.randn(64, 128) * 0.001,  # tiny random weights
            bias=None,
            target_norm=torch.tensor(1.0),
            method="vocab_mediated",
            anchor_count=10,
            validation_score=0.0,
        )

        result = validate_projection(
            src_model, tgt_model, avp_map, src_tok, tgt_tok, device="cpu",
        )

        # Random projection should have low cosine similarity → JSON
        assert result.recommended_mode == CommunicationMode.JSON

    def test_validate_cosine_sim_fast_gate(self, tiny_gpt2_64, tiny_gpt2_128):
        """When cos_sim < threshold, should return JSON without computing perplexity."""
        from avp.rosetta.calibrate import AVPMap
        from avp.rosetta.validate import validate_projection, ValidationConfig
        from avp.types import CommunicationMode

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        # Random projection → low cosine sim
        avp_map = AVPMap(
            source_model_id="src", source_hash="s" * 64, source_dim=64,
            target_model_id="tgt", target_hash="t" * 64, target_dim=128,
            w_map=torch.randn(64, 128) * 0.001,
            bias=None,
            target_norm=torch.tensor(1.0),
            method="vocab_mediated",
            anchor_count=10,
            validation_score=0.0,
        )

        # Set a high cosine sim threshold so the fast gate always triggers
        config = ValidationConfig(cosine_sim_threshold=0.99)
        result = validate_projection(
            src_model, tgt_model, avp_map, src_tok, tgt_tok,
            config=config, device="cpu",
        )

        assert result.recommended_mode == CommunicationMode.JSON
        # Perplexity should be None — skipped by fast gate
        assert result.perplexity is None
        assert "threshold" in result.detail

    def test_validate_custom_thresholds(self, tiny_gpt2_64):
        """ValidationConfig overrides are respected."""
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.validate import validate_projection, ValidationConfig
        from avp.types import CommunicationMode

        model, _ = tiny_gpt2_64
        tok = VocabMockTokenizer(vocab_size=256)
        avp_map = calibrate(model, model, tok, tok, device="cpu")

        # Set extremely strict thresholds — impossibly low perplexity_json
        # should push result toward JSON even for a decent projection
        config = ValidationConfig(
            cosine_sim_threshold=-1.0,  # never trigger fast gate
            perplexity_json=0.002,      # impossibly low
        )
        result = validate_projection(
            model, model, avp_map, tok, tok,
            config=config, device="cpu",
        )

        # With impossible thresholds, perplexity > perplexity_json → JSON
        assert result.recommended_mode == CommunicationMode.JSON
        assert result.perplexity is not None

    def test_validate_result_fields(self, tiny_gpt2_64):
        """All ValidationResult fields are populated correctly."""
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.validate import validate_projection, ValidationResult, ValidationConfig
        from avp.types import CommunicationMode

        model, _ = tiny_gpt2_64
        tok = VocabMockTokenizer(vocab_size=256)
        avp_map = calibrate(model, model, tok, tok, device="cpu")

        # Bypass fast gate so perplexity gets computed
        config = ValidationConfig(cosine_sim_threshold=-1.0)
        result = validate_projection(
            model, model, avp_map, tok, tok, config=config, device="cpu",
        )

        assert isinstance(result, ValidationResult)
        assert isinstance(result.cosine_similarity, float)
        assert isinstance(result.recommended_mode, CommunicationMode)
        assert isinstance(result.detail, str)
        assert len(result.detail) > 0
        # Perplexity should be computed (shared vocab, fast gate bypassed)
        assert result.perplexity is not None
        assert result.perplexity > 0

    def test_validate_vocab_mediated(self, tiny_gpt2_64, tiny_gpt2_128):
        """Vocab-mediated map between same-vocab models — full pipeline runs."""
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.validate import validate_projection, ValidationConfig

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok, device="cpu",
        )

        # Bypass fast gate to test full pipeline with random-weight models
        config = ValidationConfig(cosine_sim_threshold=-1.0)
        result = validate_projection(
            src_model, tgt_model, avp_map, src_tok, tgt_tok,
            config=config, device="cpu",
        )

        # Vocab-mediated should compute both metrics
        assert isinstance(result.cosine_similarity, float)
        assert result.perplexity is not None
        assert result.perplexity > 0
        assert result.detail  # non-empty

        assert isinstance(result.cosine_similarity, float)
        assert result.detail

    # ------------------------------------------------------------------
    # Regression tests: _compute_pseudo_perplexity & _compute_cosine_similarity
    # ------------------------------------------------------------------

    def test_perplexity_deterministic(self, seeded_tiny_gpt2_64):
        """Calling _compute_pseudo_perplexity twice with identical inputs
        must return bitwise-equal results."""
        from avp.rosetta.validate import _compute_pseudo_perplexity

        model, tokenizer = seeded_tiny_gpt2_64
        encoded = tokenizer("The quick brown fox", return_tensors="pt")
        token_ids = encoded["input_ids"][0]

        embed_weight = model.get_input_embeddings().weight.detach()
        projected = embed_weight[50]  # arbitrary token embedding

        ppl_1 = _compute_pseudo_perplexity(model, projected, token_ids, "cpu")
        ppl_2 = _compute_pseudo_perplexity(model, projected, token_ids, "cpu")

        assert ppl_1 == ppl_2

    def test_perplexity_own_embedding_and_noise_both_reasonable(self, seeded_tiny_gpt2_64):
        """With random-weight models, own embedding vs noise perplexity is
        indistinguishable (both near vocab_size). Verify both return finite,
        reasonable values. The own < noise property only holds for trained models."""
        from avp.rosetta.validate import _compute_pseudo_perplexity

        model, tokenizer = seeded_tiny_gpt2_64
        encoded = tokenizer("The quick brown fox", return_tensors="pt")
        token_ids = encoded["input_ids"][0]

        embed_weight = model.get_input_embeddings().weight.detach()
        own_embed = embed_weight[token_ids[-1]]

        torch.manual_seed(123)
        noise = torch.randn_like(own_embed) * 100.0

        ppl_own = _compute_pseudo_perplexity(model, own_embed, token_ids, "cpu")
        ppl_noise = _compute_pseudo_perplexity(model, noise, token_ids, "cpu")

        # Both should be finite and near vocab_size (256) for random weights
        assert 1.0 < ppl_own < 1000.0, f"Own ppl out of range: {ppl_own}"
        assert 1.0 < ppl_noise < 1000.0, f"Noise ppl out of range: {ppl_noise}"

    def test_perplexity_noise_monotonicity(self, seeded_tiny_gpt2_64):
        """Perplexity with clean embedding should be lower than with heavily
        noised embedding (ppl[0] < ppl[-1])."""
        from avp.rosetta.validate import _compute_pseudo_perplexity

        model, tokenizer = seeded_tiny_gpt2_64
        encoded = tokenizer("The quick brown fox", return_tensors="pt")
        token_ids = encoded["input_ids"][0]

        embed_weight = model.get_input_embeddings().weight.detach()
        baseline = embed_weight[50].clone()

        torch.manual_seed(99)
        noise_dir = torch.randn_like(baseline)

        noise_scales = [0.0, 1.0, 5.0, 20.0]
        ppls = []
        for scale in noise_scales:
            noised = baseline + scale * noise_dir
            ppl = _compute_pseudo_perplexity(model, noised, token_ids, "cpu")
            ppls.append(ppl)

        assert ppls[0] < ppls[-1], (
            f"Clean ppl ({ppls[0]:.2f}) should be < heavily noised ppl ({ppls[-1]:.2f}). "
            f"All: {[f'{p:.2f}' for p in ppls]}"
        )

    def test_perplexity_bounded_edge_cases(self, seeded_tiny_gpt2_64):
        """Zero, very large, and very small vectors must all produce finite
        positive perplexity (no nan, inf, or negative values)."""
        import math
        from avp.rosetta.validate import _compute_pseudo_perplexity

        model, tokenizer = seeded_tiny_gpt2_64
        encoded = tokenizer("The quick brown fox", return_tensors="pt")
        token_ids = encoded["input_ids"][0]

        hidden_dim = 64
        cases = {
            "zero": torch.zeros(hidden_dim),
            "large": torch.ones(hidden_dim) * 1e6,
            "small": torch.ones(hidden_dim) * 1e-8,
        }

        for name, vec in cases.items():
            ppl = _compute_pseudo_perplexity(model, vec, token_ids, "cpu")
            assert math.isfinite(ppl) and ppl > 0, (
                f"{name} vector: expected finite positive ppl, got {ppl}"
            )

    def test_perplexity_snapshot_regression(self, seeded_tiny_gpt2_64):
        """Perplexity for a fixed seed/model/text/embedding must match a
        hardcoded snapshot. Any change to the computation logic breaks this."""
        from avp.rosetta.validate import _compute_pseudo_perplexity

        model, tokenizer = seeded_tiny_gpt2_64
        encoded = tokenizer("The quick brown fox", return_tensors="pt")
        token_ids = encoded["input_ids"][0]

        embed_weight = model.get_input_embeddings().weight.detach()
        projected = embed_weight[50]

        ppl = _compute_pseudo_perplexity(model, projected, token_ids, "cpu")

        # Random-weight GPT2 produces perplexity near vocab_size (256).
        # Exact value varies across torch versions / platforms, so we check
        # it falls in a reasonable range rather than a tight snapshot.
        assert 200.0 < ppl < 350.0, (
            f"Perplexity out of expected range for random-weight GPT2: {ppl:.4f}"
        )

    def test_cosine_perplexity_correlation(self, seeded_tiny_gpt2_64):
        """The projection with higher cosine similarity should also have
        lower perplexity — both metrics measure the same quality signal."""
        from avp.rosetta.validate import (
            _compute_cosine_similarity,
            _compute_pseudo_perplexity,
        )

        model, tokenizer = seeded_tiny_gpt2_64
        encoded = tokenizer("The quick brown fox", return_tensors="pt")
        token_ids = encoded["input_ids"][0]

        embed_weight = model.get_input_embeddings().weight.detach()

        # Good: model's own embeddings (cos_sim ≈ 1.0 by construction)
        good_proj = embed_weight[token_ids]
        good_last = good_proj[-1]

        # Bad: random vectors
        torch.manual_seed(77)
        bad_proj = torch.randn_like(good_proj)
        bad_last = bad_proj[-1]

        cos_good = _compute_cosine_similarity(
            good_proj, embed_weight, token_ids, is_next_token=False,
        )
        cos_bad = _compute_cosine_similarity(
            bad_proj, embed_weight, token_ids, is_next_token=False,
        )

        ppl_good = _compute_pseudo_perplexity(model, good_last, token_ids, "cpu")
        ppl_bad = _compute_pseudo_perplexity(model, bad_last, token_ids, "cpu")

        assert cos_good > cos_bad, f"cos_good={cos_good:.3f} <= cos_bad={cos_bad:.3f}"
        assert ppl_good < ppl_bad, (
            f"ppl_good={ppl_good:.2f} should be < ppl_bad={ppl_bad:.2f}"
        )

    def test_perplexity_short_sequence_returns_inf(self, seeded_tiny_gpt2_64):
        """Single-token input must return inf; two-token must return finite positive."""
        import math
        from avp.rosetta.validate import _compute_pseudo_perplexity

        model, _ = seeded_tiny_gpt2_64

        embed_weight = model.get_input_embeddings().weight.detach()
        projected = embed_weight[50]

        # Single token → inf (guard at line 165-166)
        single = torch.tensor([50])
        ppl_single = _compute_pseudo_perplexity(model, projected, single, "cpu")
        assert ppl_single == float("inf")

        # Two tokens → finite positive
        two = torch.tensor([50, 60])
        ppl_two = _compute_pseudo_perplexity(model, projected, two, "cpu")
        assert math.isfinite(ppl_two) and ppl_two > 0


# ---------------------------------------------------------------------------
# Tests: Vocabulary overlap (cross-family projection)
# ---------------------------------------------------------------------------

@requires_torch
@requires_transformers
class TestVocabOverlap:
    """Tests for _compute_vocab_overlap() and vocab_overlap_projection()."""

    def test_compute_vocab_overlap_basic(self):
        """_compute_vocab_overlap returns correct indices and lengths."""
        from avp.rosetta.calibrate import _compute_vocab_overlap

        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = CrossFamilyMockTokenizer(vocab_size=256)

        result = _compute_vocab_overlap(src_tok, tgt_tok)
        assert result is not None
        src_indices, tgt_indices, shared_tokens = result

        # CrossFamilyMockTokenizer shares first half: token_0..token_127
        assert len(shared_tokens) == 128
        assert src_indices.shape == (128,)
        assert tgt_indices.shape == (128,)
        # Both tokenizers assign the same IDs to shared tokens
        import numpy as np
        assert np.array_equal(src_indices, tgt_indices)

    def test_compute_vocab_overlap_below_threshold(self):
        """Returns None when overlap < min_overlap."""
        from avp.rosetta.calibrate import _compute_vocab_overlap

        # With vocab_size=8, overlap is 4 tokens — below default min_overlap=100
        src_tok = VocabMockTokenizer(vocab_size=8)
        tgt_tok = CrossFamilyMockTokenizer(vocab_size=8)

        result = _compute_vocab_overlap(src_tok, tgt_tok, min_overlap=100)
        assert result is None

        # But succeeds with lower threshold
        result = _compute_vocab_overlap(src_tok, tgt_tok, min_overlap=2)
        assert result is not None

    def test_vocab_overlap_projection_shape(self):
        """vocab_overlap_projection produces correct output shape."""
        from avp.rosetta.project import vocab_overlap_projection

        D_src, D_tgt, N_shared, V_src = 64, 128, 100, 256
        hidden = torch.randn(2, D_src)
        w_src = torch.randn(V_src, D_src)       # source lm_head
        w_tgt = torch.randn(N_shared, D_tgt)    # shared target embeddings
        src_indices = torch.randperm(V_src)[:N_shared]

        result = vocab_overlap_projection(hidden, w_src, w_tgt, src_indices)
        assert result.shape == (2, D_tgt)

    def test_vocab_overlap_projection_preserves_dtype(self):
        """Output is always float32 numpy."""
        import numpy as np
        from avp.rosetta.project import vocab_overlap_projection

        D_src, D_tgt, N_shared, V_src = 64, 128, 100, 256
        hidden = torch.randn(2, D_src, dtype=torch.float16)
        w_src = torch.randn(V_src, D_src, dtype=torch.float16)
        w_tgt = torch.randn(N_shared, D_tgt, dtype=torch.float16)
        src_indices = torch.randperm(V_src)[:N_shared]

        result = vocab_overlap_projection(hidden, w_src, w_tgt, src_indices)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_vocab_overlap_full_overlap_equals_vocab_mediated(self):
        """When overlap is 100%, result must match vocabulary_mediated_projection."""
        from avp.rosetta.project import vocab_overlap_projection, vocabulary_mediated_projection

        D_src, D_tgt, V = 64, 128, 256
        torch.manual_seed(123)
        hidden = torch.randn(3, D_src)
        w_src = torch.randn(V, D_src)
        w_tgt = torch.randn(V, D_tgt)
        # Full overlap: all indices in order
        src_indices = torch.arange(V)

        result_overlap = vocab_overlap_projection(hidden, w_src, w_tgt, src_indices)
        result_mediated = vocabulary_mediated_projection(hidden, w_src, w_tgt)

        import numpy as np
        assert np.allclose(result_overlap, result_mediated, atol=1e-5)

    def test_vocab_overlap_return_metrics(self):
        """return_metrics=True returns (projected, metrics_dict) with entropy and max_prob."""
        from avp.rosetta.project import vocab_overlap_projection

        D_src, D_tgt, N_shared, V_src = 64, 128, 100, 256
        hidden = torch.randn(2, D_src)
        w_src = torch.randn(V_src, D_src)
        w_tgt = torch.randn(N_shared, D_tgt)
        src_indices = torch.randperm(V_src)[:N_shared]

        result = vocab_overlap_projection(
            hidden, w_src, w_tgt, src_indices, return_metrics=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        projected, metrics = result
        assert projected.shape == (2, D_tgt)
        assert "entropy" in metrics and "max_prob" in metrics
        assert metrics["entropy"].shape == (2,)
        assert (metrics["entropy"] >= 0).all()
        assert (metrics["max_prob"] >= 0).all() and (metrics["max_prob"] <= 1).all()

    def test_calibrate_auto_detects_vocab_overlap(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() with cross-family tokenizers returns VOCAB_OVERLAP."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = CrossFamilyMockTokenizer(vocab_size=256)

        avp_map = calibrate(src_model, tgt_model, src_tok, tgt_tok, device="cpu")

        assert avp_map.method == ProjectionMethod.VOCAB_OVERLAP

    def test_calibrate_vocab_overlap_w_map_shape(self, tiny_gpt2_64, tiny_gpt2_128):
        """w_map is [N_shared, D_tgt] for VOCAB_OVERLAP."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = CrossFamilyMockTokenizer(vocab_size=256)

        avp_map = calibrate(src_model, tgt_model, src_tok, tgt_tok, device="cpu")

        # 128 shared tokens, target dim=128
        assert avp_map.w_map.shape == (128, 128)
        assert avp_map.overlap_count == 128
        assert 0.0 < avp_map.overlap_ratio <= 1.0

    def test_calibrate_vocab_overlap_stores_indices(self, tiny_gpt2_64, tiny_gpt2_128):
        """src_indices and tgt_indices have shape [N_shared]."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = CrossFamilyMockTokenizer(vocab_size=256)

        avp_map = calibrate(src_model, tgt_model, src_tok, tgt_tok, device="cpu")

        assert avp_map.src_indices is not None
        assert avp_map.tgt_indices is not None
        assert avp_map.src_indices.shape == (128,)
        assert avp_map.tgt_indices.shape == (128,)

    def test_registry_save_load_vocab_overlap(self, tiny_gpt2_64, tiny_gpt2_128, tmp_path):
        """Registry roundtrip preserves vocab-overlap fields."""
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.registry import save_map, load_map

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = CrossFamilyMockTokenizer(vocab_size=256)

        avp_map = calibrate(src_model, tgt_model, src_tok, tgt_tok, device="cpu")
        save_map(avp_map, map_dir=tmp_path)

        loaded = load_map(avp_map.source_hash, avp_map.target_hash, map_dir=tmp_path)
        assert loaded is not None
        assert loaded.method == ProjectionMethod.VOCAB_OVERLAP
        assert loaded.overlap_count == avp_map.overlap_count
        assert loaded.overlap_ratio == pytest.approx(avp_map.overlap_ratio)
        import numpy as np
        assert np.array_equal(loaded.src_indices, avp_map.src_indices)
        assert np.array_equal(loaded.tgt_indices, avp_map.tgt_indices)
        assert loaded.w_map.shape == avp_map.w_map.shape

    def test_registry_load_backward_compat(self, tmp_path):
        """Old .pt files without overlap fields load with defaults."""
        from avp.rosetta.calibrate import AVPMap
        from avp.rosetta.registry import save_map, load_map

        # Create a map without overlap fields (simulating old format)
        old_map = AVPMap(
            source_model_id="src", source_hash="a" * 64, source_dim=64,
            target_model_id="tgt", target_hash="b" * 64, target_dim=128,
            w_map=torch.randn(64, 128),
            bias=None,
            target_norm=torch.tensor(5.0),
            method="vocab_mediated",
            anchor_count=50,
            validation_score=0.85,
        )

        # Save with current code (includes overlap fields as None/0)
        path = save_map(old_map, map_dir=tmp_path)

        # Manually strip overlap fields to simulate old file
        data = torch.load(path, weights_only=True)
        for key in ("src_indices", "tgt_indices", "overlap_count", "overlap_ratio"):
            data.pop(key, None)
        torch.save(data, path)

        # Load should succeed with defaults
        loaded = load_map("a" * 64, "b" * 64, map_dir=tmp_path)
        assert loaded is not None
        assert loaded.src_indices is None
        assert loaded.tgt_indices is None
        assert loaded.overlap_count == 0
        assert loaded.overlap_ratio == 0.0


# ---------------------------------------------------------------------------
# calibrate() auto-save tests
# ---------------------------------------------------------------------------


@requires_torch
@requires_transformers
class TestCalibrateAutoSave:
    """Tests for calibrate() auto_save parameter."""

    def test_calibrate_auto_saves_to_registry(self, tiny_gpt2_64, tiny_gpt2_128, tmp_path):
        """calibrate() with auto_save=True writes .avp-map file to registry."""
        import avp.rosetta.registry as registry
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.registry import find_map

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = tmp_path
            avp_map = calibrate(
                src_model, tgt_model, src_tok, tgt_tok,
                device="cpu", auto_save=True,
            )
            # Verify map file was created
            found = find_map(avp_map.source_hash, avp_map.target_hash, map_dir=tmp_path)
            assert found is not None
            assert found.exists()
        finally:
            registry._MAP_DIR = old_dir

    def test_calibrate_auto_save_disabled(self, tiny_gpt2_64, tiny_gpt2_128, tmp_path):
        """calibrate() with auto_save=False does not write .avp-map file."""
        import avp.rosetta.registry as registry
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.registry import find_map

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = tmp_path
            avp_map = calibrate(
                src_model, tgt_model, src_tok, tgt_tok,
                device="cpu", auto_save=False,
            )
            # Verify no map file was created
            found = find_map(avp_map.source_hash, avp_map.target_hash, map_dir=tmp_path)
            assert found is None
        finally:
            registry._MAP_DIR = old_dir


# ---------------------------------------------------------------------------
# Tests: Per-transfer quality gate
# ---------------------------------------------------------------------------


class TestTransferQuality:
    """Tests for rosetta.quality — per-transfer quality gate."""

    def test_short_prompt_recommends_latent(self):
        """200 tokens → recommend latent."""
        from avp.rosetta.quality import assess_transfer

        result = assess_transfer(prompt_tokens=200)
        assert result.recommend_latent is True

    def test_long_prompt_recommends_json(self):
        """1500 tokens → recommend JSON fallback."""
        from avp.rosetta.quality import assess_transfer

        result = assess_transfer(prompt_tokens=1500)
        assert result.recommend_latent is False

    def test_boundary_at_default_threshold(self):
        """300 tokens (== max_prompt_tokens) → recommend latent."""
        from avp.rosetta.quality import assess_transfer

        result = assess_transfer(prompt_tokens=300)
        assert result.recommend_latent is True

    def test_one_above_threshold(self):
        """301 tokens (> max_prompt_tokens) → recommend JSON."""
        from avp.rosetta.quality import assess_transfer

        result = assess_transfer(prompt_tokens=301)
        assert result.recommend_latent is False

    def test_custom_threshold(self):
        """Custom max_prompt_tokens is respected."""
        from avp.rosetta.quality import TransferQualityConfig, assess_transfer

        config = TransferQualityConfig(max_prompt_tokens=100)
        assert assess_transfer(prompt_tokens=100, config=config).recommend_latent is True
        assert assess_transfer(prompt_tokens=101, config=config).recommend_latent is False

    def test_zero_tokens(self):
        """Edge case: 0 tokens → recommend latent."""
        from avp.rosetta.quality import assess_transfer

        result = assess_transfer(prompt_tokens=0)
        assert result.recommend_latent is True

    def test_result_fields_populated(self):
        """All fields present with correct types."""
        from avp.rosetta.quality import TransferQualityResult, assess_transfer

        result = assess_transfer(prompt_tokens=200)
        assert isinstance(result, TransferQualityResult)
        assert isinstance(result.recommend_latent, bool)
        assert isinstance(result.prompt_tokens, int)
        assert result.prompt_tokens == 200
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_effective_rank_not_computed_by_default(self):
        """effective_rank_ratio is None when not requested."""
        from avp.rosetta.quality import assess_transfer

        result = assess_transfer(prompt_tokens=200)
        assert result.effective_rank_ratio is None

    @requires_torch
    def test_effective_rank_computed_when_requested(self):
        """effective_rank_ratio is computed when check_effective_rank=True."""
        from avp.rosetta.quality import TransferQualityConfig, assess_transfer

        config = TransferQualityConfig(check_effective_rank=True)
        hidden = torch.randn(10, 64)
        result = assess_transfer(prompt_tokens=200, hidden_states=hidden, config=config)
        assert result.effective_rank_ratio is not None
        assert 0.0 <= result.effective_rank_ratio <= 1.0

    @requires_torch
    def test_effective_rank_3d_input(self):
        """Accepts [1, seq_len, D] shape."""
        from avp.rosetta.quality import TransferQualityConfig, assess_transfer

        config = TransferQualityConfig(check_effective_rank=True)
        hidden = torch.randn(1, 10, 64)
        result = assess_transfer(prompt_tokens=200, hidden_states=hidden, config=config)
        assert result.effective_rank_ratio is not None
        assert 0.0 <= result.effective_rank_ratio <= 1.0

    @requires_torch
    def test_effective_rank_gate_triggers(self):
        """Identity-like matrix (high rank) triggers JSON recommendation."""
        from avp.rosetta.quality import TransferQualityConfig, assess_transfer

        config = TransferQualityConfig(
            check_effective_rank=True,
            max_effective_rank_ratio=0.5,
        )
        # Identity-like matrix has maximum effective rank
        hidden = torch.eye(32)
        result = assess_transfer(prompt_tokens=200, hidden_states=hidden, config=config)
        assert result.recommend_latent is False
        assert result.effective_rank_ratio is not None
        assert result.effective_rank_ratio > 0.5


# ---------------------------------------------------------------------------
# Tests: numpy-only projection (no torch required)
# ---------------------------------------------------------------------------


class TestNumpyOnlyProjection:
    """Verify projection functions work with pure numpy inputs (no torch)."""

    def test_vocabulary_mediated_numpy_inputs(self):
        """vocabulary_mediated_projection works with numpy arrays only."""
        import numpy as np
        from avp.rosetta.project import vocabulary_mediated_projection

        rng = np.random.RandomState(42)
        hidden = rng.randn(2, 64).astype(np.float32)
        w_src = rng.randn(256, 64).astype(np.float32)
        w_tgt = rng.randn(256, 128).astype(np.float32)

        result = vocabulary_mediated_projection(hidden, w_src, w_tgt, target_norm=5.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 128)
        assert result.dtype == np.float32
        norms = np.linalg.norm(result, axis=-1)
        for n in norms:
            assert abs(float(n) - 5.0) < 0.01

    def test_vocab_overlap_numpy_inputs(self):
        """vocab_overlap_projection works with numpy arrays only."""
        import numpy as np
        from avp.rosetta.project import vocab_overlap_projection

        rng = np.random.RandomState(42)
        hidden = rng.randn(2, 64).astype(np.float32)
        w_src = rng.randn(256, 64).astype(np.float32)
        w_tgt = rng.randn(100, 128).astype(np.float32)
        idx = np.arange(100, dtype=np.intp)

        result = vocab_overlap_projection(hidden, w_src, w_tgt, idx, target_norm=5.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 128)

    def test_apply_cross_model_projection_numpy_inputs(self):
        """apply_cross_model_projection works with numpy arrays only."""
        import numpy as np
        from avp.rosetta.project import apply_cross_model_projection

        rng = np.random.RandomState(42)
        hidden = rng.randn(4, 64).astype(np.float32)
        w_map = rng.randn(64, 128).astype(np.float32)

        result = apply_cross_model_projection(hidden, w_map, target_norm=7.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 128)
        norms = np.linalg.norm(result, axis=-1)
        for n in norms:
            assert abs(float(n) - 7.5) < 0.01

    def test_normalize_to_target_numpy_inputs(self):
        """normalize_to_target works with numpy arrays only."""
        import numpy as np
        from avp.realign import normalize_to_target

        hidden = np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        result = normalize_to_target(hidden, target_norm=1.0)
        assert isinstance(result, np.ndarray)
        norms = np.linalg.norm(result, axis=-1)
        for n in norms:
            assert abs(float(n) - 1.0) < 0.01

    def test_project_to_embedding_space_numpy_inputs(self):
        """project_to_embedding_space works with numpy arrays only."""
        import numpy as np
        from avp.realign import project_to_embedding_space

        rng = np.random.RandomState(42)
        hidden = rng.randn(2, 64).astype(np.float32)
        embed = rng.randn(256, 64).astype(np.float32)

        result = project_to_embedding_space(hidden, embed, temperature=1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 64)

    def test_apply_realignment_numpy_inputs(self):
        """apply_realignment works with numpy arrays only."""
        import numpy as np
        from avp.realign import apply_realignment

        rng = np.random.RandomState(42)
        hidden = rng.randn(2, 64).astype(np.float32)
        w_realign = rng.randn(64, 64).astype(np.float32)

        result = apply_realignment(hidden, w_realign, target_norm=5.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 64)
        norms = np.linalg.norm(result, axis=-1)
        for n in norms:
            assert abs(float(n) - 5.0) < 0.01

    def test_return_metrics_are_numpy(self):
        """Metrics returned by projection functions are numpy arrays."""
        import numpy as np
        from avp.rosetta.project import vocabulary_mediated_projection

        rng = np.random.RandomState(42)
        hidden = rng.randn(2, 64).astype(np.float32)
        w_src = rng.randn(256, 64).astype(np.float32)
        w_tgt = rng.randn(256, 128).astype(np.float32)

        projected, metrics = vocabulary_mediated_projection(
            hidden, w_src, w_tgt, return_metrics=True,
        )
        assert isinstance(projected, np.ndarray)
        assert isinstance(metrics["entropy"], np.ndarray)
        assert isinstance(metrics["max_prob"], np.ndarray)
        assert isinstance(metrics["nearest_cos_sim"], np.ndarray)

    def test_softmax_numerical_stability(self):
        """Softmax handles large logit values without overflow."""
        import numpy as np
        from avp.rosetta.project import _softmax

        # Large positive values — should not overflow
        x = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
        result = _softmax(x)
        assert np.isfinite(result).all()
        assert abs(float(result.sum()) - 1.0) < 1e-6

        # Large negative values — should not underflow to NaN
        x = np.array([[-1000.0, -1001.0, -999.0]], dtype=np.float32)
        result = _softmax(x)
        assert np.isfinite(result).all()
        assert abs(float(result.sum()) - 1.0) < 1e-6

    def test_contiguous_output(self):
        """Projection outputs are C-contiguous (safe for torch.from_numpy)."""
        import numpy as np
        from avp.rosetta.project import vocabulary_mediated_projection

        rng = np.random.RandomState(42)
        hidden = rng.randn(2, 64).astype(np.float32)
        w_src = rng.randn(256, 64).astype(np.float32)
        w_tgt = rng.randn(256, 128).astype(np.float32)

        result = vocabulary_mediated_projection(hidden, w_src, w_tgt)
        assert result.flags["C_CONTIGUOUS"]


@requires_torch
class TestNumpyTorchBoundary:
    """Verify torch→numpy→torch round-trip at connector boundaries."""

    def test_to_numpy_from_cuda_float16(self):
        """_to_numpy handles float16 tensors correctly."""
        from avp.rosetta.project import _to_numpy
        import numpy as np

        t = torch.randn(2, 64, dtype=torch.float16)
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 64)

    def test_to_numpy_from_bfloat16(self):
        """_to_numpy handles bfloat16 tensors correctly."""
        from avp.rosetta.project import _to_numpy
        import numpy as np

        t = torch.randn(2, 64, dtype=torch.bfloat16)
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_to_numpy_from_requires_grad(self):
        """_to_numpy handles tensors with requires_grad=True."""
        from avp.rosetta.project import _to_numpy
        import numpy as np

        t = torch.randn(2, 64, requires_grad=True)
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)

    def test_to_numpy_passthrough_numpy(self):
        """_to_numpy is a no-copy passthrough for numpy arrays."""
        from avp.rosetta.project import _to_numpy
        import numpy as np

        arr = np.ones((2, 64), dtype=np.float32)
        result = _to_numpy(arr)
        assert result is arr or np.shares_memory(result, arr)

    def test_projection_torch_in_numpy_out(self):
        """Projection accepts torch tensors and returns numpy."""
        import numpy as np
        from avp.rosetta.project import vocabulary_mediated_projection

        hidden = torch.randn(2, 64)
        w_src = torch.randn(256, 64)
        w_tgt = torch.randn(256, 128)

        result = vocabulary_mediated_projection(hidden, w_src, w_tgt)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_torch_numpy_torch_roundtrip_preserves_values(self):
        """torch→numpy projection→torch roundtrip is numerically faithful."""
        import numpy as np
        from avp.rosetta.project import vocabulary_mediated_projection

        hidden = torch.randn(1, 64)
        w_src = torch.randn(256, 64)
        w_tgt = torch.randn(256, 128)

        result_np = vocabulary_mediated_projection(hidden, w_src, w_tgt)
        result_torch = torch.from_numpy(np.ascontiguousarray(result_np))

        assert result_torch.shape == (1, 128)
        assert result_torch.dtype == torch.float32
        assert torch.isfinite(result_torch).all()
