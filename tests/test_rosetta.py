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
        return {f"token_{i}": i for i in range(self.vocab_size)}


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
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(4, 64)
        w_map = torch.randn(64, 128)
        target_norm = torch.tensor(7.5)

        result = apply_cross_model_projection(hidden, w_map, target_norm)
        norms = result.norm(dim=-1)
        for n in norms:
            assert abs(n.item() - 7.5) < 0.01

    def test_projection_with_bias(self):
        """Bias is correctly applied."""
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(2, 64)
        w_map = torch.randn(64, 128)
        bias = torch.randn(128)
        target_norm = torch.tensor(5.0)

        result = apply_cross_model_projection(hidden, w_map, target_norm, bias=bias)
        assert result.shape == (2, 128)
        # Result should differ from no-bias version
        result_no_bias = apply_cross_model_projection(hidden, w_map, target_norm)
        assert not torch.allclose(result, result_no_bias)

    def test_projection_preserves_dtype(self):
        """Output dtype matches input dtype."""
        from avp.rosetta.project import apply_cross_model_projection

        hidden = torch.randn(2, 64, dtype=torch.float16)
        w_map = torch.randn(64, 128)
        target_norm = torch.tensor(5.0)

        result = apply_cross_model_projection(hidden, w_map, target_norm)
        assert result.dtype == torch.float16

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

    def test_calibrate_ridge(self, tiny_gpt2_64, tiny_gpt2_128):
        """Ridge regression when dims differ."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_128

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            anchor_texts=self.ANCHOR_TEXTS, device="cpu",
        )

        assert avp_map.method == ProjectionMethod.RIDGE
        assert avp_map.w_map.shape == (64, 128)
        assert avp_map.source_dim == 64
        assert avp_map.target_dim == 128
        assert avp_map.bias is None
        assert avp_map.anchor_count == 8  # 10 - 20% validation
        assert isinstance(avp_map.validation_score, float)

    def test_calibrate_procrustes(self, tiny_gpt2_64, tiny_gpt2_64_v2):
        """Procrustes when dims match."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_64_v2

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            anchor_texts=self.ANCHOR_TEXTS, device="cpu",
        )

        assert avp_map.method == ProjectionMethod.PROCRUSTES
        assert avp_map.w_map.shape == (64, 64)

        # Procrustes matrix should be approximately orthogonal: W^T @ W ≈ I
        product = avp_map.w_map.T @ avp_map.w_map
        identity = torch.eye(64)
        assert torch.allclose(product, identity, atol=1e-5)

    def test_calibrate_auto_selects_ridge(self, tiny_gpt2_64, tiny_gpt2_128):
        """Auto method selects ridge when dims differ."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_128

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            anchor_texts=self.ANCHOR_TEXTS, method="auto", device="cpu",
        )
        assert avp_map.method == ProjectionMethod.RIDGE

    def test_calibrate_auto_selects_procrustes(self, tiny_gpt2_64, tiny_gpt2_64_v2):
        """Auto method selects procrustes when dims match."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_64_v2

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            anchor_texts=self.ANCHOR_TEXTS, method="auto", device="cpu",
        )
        assert avp_map.method == ProjectionMethod.PROCRUSTES

    def test_calibrate_procrustes_fails_different_dims(self, tiny_gpt2_64, tiny_gpt2_128):
        """Procrustes raises ValueError when dims differ."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_128

        with pytest.raises(ValueError, match="Procrustes requires same dimensions"):
            calibrate(
                src_model, tgt_model, src_tok, tgt_tok,
                anchor_texts=self.ANCHOR_TEXTS, method="procrustes", device="cpu",
            )

    def test_calibrate_too_few_anchors(self, tiny_gpt2_64, tiny_gpt2_128):
        """Calibration fails with too few anchor texts."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_128

        with pytest.raises(ValueError, match="at least 5"):
            calibrate(
                src_model, tgt_model, src_tok, tgt_tok,
                anchor_texts=["a", "b", "c"], device="cpu",
            )

    def test_calibrate_target_norm(self, tiny_gpt2_64, tiny_gpt2_128):
        """Calibration computes a valid target norm."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_128

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            anchor_texts=self.ANCHOR_TEXTS, device="cpu",
        )
        assert avp_map.target_norm.item() > 0


# ---------------------------------------------------------------------------
# Tests: registry (save / load / find)
# ---------------------------------------------------------------------------

@requires_torch
@requires_transformers
class TestRegistry:
    def test_save_load_roundtrip(self, tmp_path, tiny_gpt2_64, tiny_gpt2_128):
        """Save and load produces identical AVPMap."""
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.registry import load_map, save_map

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_128

        anchor_texts = [
            "Test sentence one.", "Test sentence two.",
            "Test sentence three.", "Test sentence four.",
            "Test sentence five.", "Test sentence six.",
        ]
        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            anchor_texts=anchor_texts, device="cpu",
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
        from avp.rosetta.registry import _map_filename, _MAP_DIR
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
            method="ridge",
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
        from avp.rosetta.project import vocabulary_mediated_projection

        vocab_size = 256
        d_src, d_tgt = 64, 128
        source_lm_head = torch.randn(vocab_size, d_src)
        target_embed = torch.randn(vocab_size, d_tgt)

        # Use a real embedding as input (high confidence in one token)
        hidden = source_lm_head[42].unsqueeze(0)  # [1, D_src] — looks like token 42
        result = vocabulary_mediated_projection(hidden, source_lm_head, target_embed)

        # Result should be close to target_embed[42] since softmax
        # should peak at token 42
        result_norm = result / result.norm(dim=-1, keepdim=True)
        tgt_norm = target_embed[42:43] / target_embed[42:43].norm(dim=-1, keepdim=True)
        cos_sim = (result_norm * tgt_norm).sum().item()
        assert cos_sim > 0.9, f"Cosine similarity {cos_sim:.3f} should be > 0.9"

    def test_vocab_mediated_matches_same_model(self):
        """When source==target weights, matches project_to_embedding_space()."""
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

        assert torch.allclose(result_vocab, result_same, atol=1e-5)

    def test_vocab_mediated_preserves_dtype(self):
        """Output dtype matches input dtype."""
        from avp.rosetta.project import vocabulary_mediated_projection

        hidden = torch.randn(2, 64, dtype=torch.float16)
        source_lm_head = torch.randn(256, 64)
        target_embed = torch.randn(256, 128)

        result = vocabulary_mediated_projection(hidden, source_lm_head, target_embed)
        assert result.dtype == torch.float16


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

        tok = MockTokenizer(vocab_size=256)  # no get_vocab()
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

    def test_calibrate_falls_back_to_ridge(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() falls back to ridge when vocab differs."""
        from avp.rosetta.calibrate import calibrate

        src_model, src_tok = tiny_gpt2_64
        tgt_model, tgt_tok = tiny_gpt2_128
        # MockTokenizer has no get_vocab() → not shared → ridge fallback

        anchor_texts = [
            "Test one.", "Test two.", "Test three.",
            "Test four.", "Test five.", "Test six.",
        ]
        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            anchor_texts=anchor_texts, device="cpu",
        )
        assert avp_map.method == ProjectionMethod.RIDGE

    def test_calibrate_vocab_mediated_explicit(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() with method='vocab_mediated' works explicitly."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok,
            method="vocab_mediated", device="cpu",
        )
        assert avp_map.method == ProjectionMethod.VOCAB_MEDIATED

    def test_calibrate_vocab_mediated_fails_different_vocab(self, tiny_gpt2_64, tiny_gpt2_128):
        """calibrate() with method='vocab_mediated' fails when vocab differs."""
        from avp.rosetta.calibrate import calibrate

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=128)

        with pytest.raises(ValueError, match="same vocabulary"):
            calibrate(
                src_model, tgt_model, src_tok, tgt_tok,
                method="vocab_mediated", device="cpu",
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
        from avp.rosetta.calibrate import AVPMap, calibrate
        from avp.rosetta.validate import validate_projection, ValidationConfig
        from avp.types import CommunicationMode

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
            method="ridge",
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
            method="ridge",
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

        # Set extremely strict thresholds — perplexity_latent=0 should
        # push result toward HYBRID or JSON even for a decent projection
        config = ValidationConfig(
            cosine_sim_threshold=-1.0,  # never trigger fast gate
            perplexity_latent=0.001,    # impossibly low
            perplexity_json=0.002,      # also impossibly low
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
        from avp.types import CommunicationMode

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

    def test_connector_validate_cross_model(self, tiny_gpt2_64, tiny_gpt2_128):
        """HuggingFaceConnector.validate_cross_model() convenience method works."""
        from avp.connectors.huggingface import HuggingFaceConnector
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.validate import ValidationResult

        src_model, _ = tiny_gpt2_64
        tgt_model, _ = tiny_gpt2_128
        src_tok = VocabMockTokenizer(vocab_size=256)
        tgt_tok = VocabMockTokenizer(vocab_size=256)

        src_connector = HuggingFaceConnector(
            model=src_model, tokenizer=src_tok, device="cpu",
        )
        tgt_connector = HuggingFaceConnector(
            model=tgt_model, tokenizer=tgt_tok, device="cpu",
        )

        avp_map = calibrate(
            src_model, tgt_model, src_tok, tgt_tok, device="cpu",
        )

        result = src_connector.validate_cross_model(tgt_connector, avp_map)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.cosine_similarity, float)
        assert result.detail
