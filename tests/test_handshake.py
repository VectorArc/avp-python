"""Tests for AVP handshake negotiation."""

import importlib.util
import logging
import tempfile

import pytest

from avp.handshake import (
    CompatibilityResolver,
    HelloMessage,
    compute_model_hash,
    extract_model_identity,
)
from avp.types import CommunicationMode, ModelIdentity

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)


# --- HelloMessage ---


def test_hello_message_roundtrip():
    identity = ModelIdentity(
        model_family="llama",
        model_id="llama-7b",
        model_hash="abc",
        hidden_dim=4096,
        num_layers=32,
    )
    hello = HelloMessage(
        agent_id="agent-1",
        identity=identity,
        capabilities={"latent": True},
    )
    d = hello.to_dict()
    restored = HelloMessage.from_dict(d)

    assert restored.agent_id == "agent-1"
    assert restored.avp_version == "0.2.0"
    assert restored.identity.model_family == "llama"
    assert restored.identity.hidden_dim == 4096
    assert restored.capabilities["latent"] is True


def test_hello_message_without_identity():
    hello = HelloMessage(agent_id="no-model")
    d = hello.to_dict()
    assert "identity" not in d

    restored = HelloMessage.from_dict(d)
    assert restored.identity is None


# --- compute_model_hash ---


def test_model_hash_deterministic():
    config = {"hidden_size": 4096, "num_hidden_layers": 32, "model_type": "llama"}
    h1 = compute_model_hash(config)
    h2 = compute_model_hash(config)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_model_hash_different_configs():
    c1 = {"hidden_size": 4096, "model_type": "llama"}
    c2 = {"hidden_size": 2048, "model_type": "llama"}
    assert compute_model_hash(c1) != compute_model_hash(c2)


def test_model_hash_key_order_independent():
    c1 = {"b": 2, "a": 1}
    c2 = {"a": 1, "b": 2}
    assert compute_model_hash(c1) == compute_model_hash(c2)


# --- extract_model_identity ---


def test_extract_from_dict(model_config_dict):
    identity = extract_model_identity(model_config_dict)
    assert identity.model_family == "llama"
    assert identity.model_id == "meta-llama/Llama-2-7b"
    assert identity.hidden_dim == 4096
    assert identity.num_layers == 32
    assert identity.num_kv_heads == 32
    assert identity.head_dim == 128
    assert len(identity.model_hash) == 64


def test_extract_from_unsupported_type():
    with pytest.raises(Exception):
        extract_model_identity(42)


# --- CompatibilityResolver ---


def test_resolve_same_hash():
    """Same model_hash → LATENT."""
    local = ModelIdentity(model_hash="aaa", model_family="llama", hidden_dim=4096, num_layers=32)
    remote = ModelIdentity(model_hash="aaa", model_family="llama", hidden_dim=4096, num_layers=32)
    result = CompatibilityResolver.resolve(local, remote)

    assert result.mode == CommunicationMode.LATENT
    assert len(result.session_id) > 0


def test_resolve_same_structure():
    """Same family + hidden_dim + num_layers → LATENT (even if hash differs)."""
    local = ModelIdentity(model_hash="aaa", model_family="llama", hidden_dim=4096, num_layers=32)
    remote = ModelIdentity(model_hash="bbb", model_family="llama", hidden_dim=4096, num_layers=32)
    result = CompatibilityResolver.resolve(local, remote)

    assert result.mode == CommunicationMode.LATENT


def test_resolve_different_family():
    """Different model_family → JSON."""
    local = ModelIdentity(model_hash="aaa", model_family="llama", hidden_dim=4096, num_layers=32)
    remote = ModelIdentity(model_hash="bbb", model_family="qwen", hidden_dim=4096, num_layers=32)
    result = CompatibilityResolver.resolve(local, remote)

    assert result.mode == CommunicationMode.JSON


def test_resolve_different_dimensions():
    """Same family but different hidden_dim → JSON."""
    local = ModelIdentity(model_family="llama", hidden_dim=4096, num_layers=32)
    remote = ModelIdentity(model_family="llama", hidden_dim=2048, num_layers=32)
    result = CompatibilityResolver.resolve(local, remote)

    assert result.mode == CommunicationMode.JSON


def test_resolve_empty_identities():
    """Empty identities → JSON."""
    result = CompatibilityResolver.resolve(ModelIdentity(), ModelIdentity())
    assert result.mode == CommunicationMode.JSON


def test_resolve_returns_both_identities():
    local = ModelIdentity(model_family="llama")
    remote = ModelIdentity(model_family="qwen")
    result = CompatibilityResolver.resolve(local, remote)

    assert result.local_identity is local
    assert result.remote_identity is remote


def test_resolve_same_family_different_layers():
    """Same family + hidden_dim but different num_layers → JSON (not structural match)."""
    local = ModelIdentity(
        model_hash="aaa", model_family="qwen2", hidden_dim=896, num_layers=24,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="qwen2", hidden_dim=896, num_layers=28,
    )
    result = CompatibilityResolver.resolve(local, remote)
    assert result.mode == CommunicationMode.JSON
    assert result.avp_map_id == ""


# --- Resolution priority chain ---


def test_resolve_priority_hash_beats_structural():
    """Rule 1 (hash match) wins over rule 2 (structural match).

    Both rules would match, but hash match should resolve LATENT without
    setting avp_map_id (it's the same model, not cross-model).
    """
    same_hash = "h" * 64
    tok_hash = "t" * 64
    local = ModelIdentity(
        model_hash=same_hash, model_family="qwen2",
        hidden_dim=896, num_layers=24, tokenizer_hash=tok_hash,
    )
    remote = ModelIdentity(
        model_hash=same_hash, model_family="qwen2",
        hidden_dim=896, num_layers=24, tokenizer_hash=tok_hash,
    )
    result = CompatibilityResolver.resolve(local, remote)
    assert result.mode == CommunicationMode.LATENT
    assert result.avp_map_id == ""  # no vocab/map needed — identical models


def test_resolve_priority_structural_beats_tokenizer():
    """Rule 2 (structural match) wins over rule 3 (shared tokenizer).

    Same family+dim+layers should resolve as LATENT without avp_map_id,
    even when tokenizer_hash also matches. No vocab-mediated projection
    needed for structurally identical models.
    """
    tok_hash = "t" * 64
    local = ModelIdentity(
        model_hash="aaa", model_family="qwen2",
        hidden_dim=896, num_layers=24, tokenizer_hash=tok_hash,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="qwen2",
        hidden_dim=896, num_layers=24, tokenizer_hash=tok_hash,
    )
    result = CompatibilityResolver.resolve(local, remote)
    assert result.mode == CommunicationMode.LATENT
    assert result.avp_map_id == ""  # structural match, no vocab needed


def test_resolve_priority_tokenizer_beats_map_file():
    """Rule 3 (shared tokenizer) wins over rule 4 (.avp-map file).

    When both tokenizer match and a map file exists, tokenizer should
    win because it's checked first and doesn't require file I/O.
    """
    import avp.rosetta.registry as registry
    from avp.rosetta.registry import _map_filename

    tok_hash = "t" * 64
    src_hash = "src" + "0" * 61
    tgt_hash = "tgt" + "0" * 61
    local = ModelIdentity(
        model_hash=src_hash, model_family="qwen2",
        hidden_dim=1536, num_layers=28, tokenizer_hash=tok_hash,
    )
    remote = ModelIdentity(
        model_hash=tgt_hash, model_family="qwen2",
        hidden_dim=896, num_layers=24, tokenizer_hash=tok_hash,
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        # Create a map file that would be found by rule 4
        filename = _map_filename(src_hash, tgt_hash)
        (tmp_path / filename).write_bytes(b"dummy")

        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = tmp_path
            result = CompatibilityResolver.resolve(local, remote)
            assert result.mode == CommunicationMode.LATENT
            # Tokenizer wins: avp_map_id should be "vocab:..." not the file map id
            assert result.avp_map_id.startswith("vocab:")
        finally:
            registry._MAP_DIR = old_dir


def test_resolve_map_file_beats_json():
    """Rule 4 (.avp-map file) wins over rule 5 (JSON fallback).

    Different family, no tokenizer hash, but map file exists → LATENT.
    """
    import avp.rosetta.registry as registry
    from avp.rosetta.registry import _map_filename, map_id

    src_hash = "src" + "0" * 61
    tgt_hash = "tgt" + "0" * 61
    local = ModelIdentity(
        model_hash=src_hash, model_family="gpt2",
        hidden_dim=64, num_layers=2,
    )
    remote = ModelIdentity(
        model_hash=tgt_hash, model_family="llama",
        hidden_dim=128, num_layers=4,
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        filename = _map_filename(src_hash, tgt_hash)
        (tmp_path / filename).write_bytes(b"dummy")

        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = tmp_path
            result = CompatibilityResolver.resolve(local, remote)
            assert result.mode == CommunicationMode.LATENT
            assert result.avp_map_id == map_id(src_hash, tgt_hash)
        finally:
            registry._MAP_DIR = old_dir


def test_resolve_all_rules_miss_gives_json():
    """When no rule matches, result is JSON with empty avp_map_id.

    Different family, different dims, no tokenizer hash, no map file.
    """
    import avp.rosetta.registry as registry

    local = ModelIdentity(
        model_hash="aaa", model_family="gpt2",
        hidden_dim=64, num_layers=2,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="llama",
        hidden_dim=128, num_layers=4,
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = pathlib.Path(tmp)
            result = CompatibilityResolver.resolve(local, remote)
            assert result.mode == CommunicationMode.JSON
            assert result.avp_map_id == ""
        finally:
            registry._MAP_DIR = old_dir


def test_resolve_tokenizer_fallthrough_on_dim_mismatch():
    """Same family, different dims, but shared tokenizer → LATENT via tokenizer (rule 3).

    Rules 1-2 miss (different hash, different dims). Rule 3 catches
    the shared tokenizer and sets avp_map_id="vocab:...".
    """
    tok_hash = "t" * 64
    local = ModelIdentity(
        model_hash="aaa", model_family="qwen2",
        hidden_dim=1536, num_layers=28, tokenizer_hash=tok_hash,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="qwen2",
        hidden_dim=896, num_layers=24, tokenizer_hash=tok_hash,
    )
    result = CompatibilityResolver.resolve(local, remote)
    assert result.mode == CommunicationMode.LATENT
    assert result.avp_map_id.startswith("vocab:")


def test_resolve_avp_map_id_format():
    """avp_map_id for vocab-mediated uses the format 'vocab:{hash[:16]}'."""
    tok_hash = "abcdef1234567890" + "x" * 48
    local = ModelIdentity(
        model_hash="aaa", model_family="qwen2",
        hidden_dim=1536, num_layers=28, tokenizer_hash=tok_hash,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="qwen2",
        hidden_dim=896, num_layers=24, tokenizer_hash=tok_hash,
    )
    result = CompatibilityResolver.resolve(local, remote)
    assert result.avp_map_id == "vocab:abcdef1234567890"


def test_resolve_no_tokenizer_hash_skips_rule3():
    """Empty tokenizer_hash on one side skips rule 3, even if models share family.

    If only one side has a tokenizer_hash, rule 3 should not match.
    """
    import avp.rosetta.registry as registry

    tok_hash = "t" * 64
    local = ModelIdentity(
        model_hash="aaa", model_family="qwen2",
        hidden_dim=1536, num_layers=28, tokenizer_hash=tok_hash,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="qwen2",
        hidden_dim=896, num_layers=24, tokenizer_hash="",  # no hash
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = pathlib.Path(tmp)
            result = CompatibilityResolver.resolve(local, remote)
            assert result.mode == CommunicationMode.JSON
            assert result.avp_map_id == ""
        finally:
            registry._MAP_DIR = old_dir


# --- Handshake logging ---


def test_resolve_logs_hash_match(caplog):
    """Verify hash_match resolution is logged."""
    local = ModelIdentity(model_hash="aaa", model_family="llama", hidden_dim=4096, num_layers=32)
    remote = ModelIdentity(model_hash="aaa", model_family="llama", hidden_dim=4096, num_layers=32)
    with caplog.at_level(logging.DEBUG, logger="avp.handshake"):
        CompatibilityResolver.resolve(local, remote)
    assert any("hash_match" in r.message for r in caplog.records)


# --- Vocabulary overlap discovery (rule 5) ---


class _OverlapTokenizerA:
    """Mock tokenizer A: tokens token_0..token_199."""

    def get_vocab(self):
        return {f"token_{i}": i for i in range(200)}


class _OverlapTokenizerB:
    """Mock tokenizer B: token_0..token_149 shared, alt_150..alt_199 unique."""

    def get_vocab(self):
        vocab = {}
        for i in range(200):
            if i < 150:
                vocab[f"token_{i}"] = i
            else:
                vocab[f"alt_{i}"] = i
        return vocab


class _TinyTokenizer:
    """Mock tokenizer with only 50 tokens (below min_overlap=100)."""

    def get_vocab(self):
        return {f"token_{i}": i for i in range(50)}


def test_resolve_vocab_overlap_with_tokenizers():
    """Rule 5: tokenizer objects with partial overlap → LATENT with vocab_overlap."""
    import avp.rosetta.registry as registry

    local = ModelIdentity(
        model_hash="aaa", model_family="qwen2",
        hidden_dim=1536, num_layers=28,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="llama",
        hidden_dim=2048, num_layers=32,
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = pathlib.Path(tmp)
            result = CompatibilityResolver.resolve(
                local, remote,
                source_tokenizer=_OverlapTokenizerA(),
                target_tokenizer=_OverlapTokenizerB(),
            )
            assert result.mode == CommunicationMode.LATENT
            assert result.avp_map_id.startswith("vocab_overlap:")
            assert "150" in result.avp_map_id  # 150 shared tokens
        finally:
            registry._MAP_DIR = old_dir


def test_resolve_no_tokenizers_falls_to_json():
    """Without tokenizer objects, rule 5 is skipped → JSON (backward compat)."""
    import avp.rosetta.registry as registry

    local = ModelIdentity(
        model_hash="aaa", model_family="qwen2",
        hidden_dim=1536, num_layers=28,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="llama",
        hidden_dim=2048, num_layers=32,
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = pathlib.Path(tmp)
            result = CompatibilityResolver.resolve(local, remote)
            assert result.mode == CommunicationMode.JSON
            assert result.avp_map_id == ""
        finally:
            registry._MAP_DIR = old_dir


def test_resolve_low_overlap_falls_to_json():
    """Tokenizers with <100 shared tokens → overlap below threshold → JSON."""
    import avp.rosetta.registry as registry

    local = ModelIdentity(
        model_hash="aaa", model_family="qwen2",
        hidden_dim=1536, num_layers=28,
    )
    remote = ModelIdentity(
        model_hash="bbb", model_family="llama",
        hidden_dim=2048, num_layers=32,
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = pathlib.Path(tmp)
            result = CompatibilityResolver.resolve(
                local, remote,
                source_tokenizer=_OverlapTokenizerA(),
                target_tokenizer=_TinyTokenizer(),
            )
            assert result.mode == CommunicationMode.JSON
            assert result.avp_map_id == ""
        finally:
            registry._MAP_DIR = old_dir


def test_resolve_vocab_overlap_lower_priority_than_map_file():
    """Rule 4 (.avp-map file) wins over rule 5 (vocab overlap)."""
    import avp.rosetta.registry as registry
    from avp.rosetta.registry import _map_filename, map_id

    src_hash = "src" + "0" * 61
    tgt_hash = "tgt" + "0" * 61
    local = ModelIdentity(
        model_hash=src_hash, model_family="qwen2",
        hidden_dim=1536, num_layers=28,
    )
    remote = ModelIdentity(
        model_hash=tgt_hash, model_family="llama",
        hidden_dim=2048, num_layers=32,
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        filename = _map_filename(src_hash, tgt_hash)
        (tmp_path / filename).write_bytes(b"dummy")

        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = tmp_path
            result = CompatibilityResolver.resolve(
                local, remote,
                source_tokenizer=_OverlapTokenizerA(),
                target_tokenizer=_OverlapTokenizerB(),
            )
            assert result.mode == CommunicationMode.LATENT
            # Rule 4 wins: avp_map_id is the file-based map id, not vocab_overlap
            assert result.avp_map_id == map_id(src_hash, tgt_hash)
        finally:
            registry._MAP_DIR = old_dir


# --- Handshake logging ---


def test_resolve_logs_json_fallback(caplog):
    """Verify json_fallback resolution is logged."""
    import avp.rosetta.registry as registry

    local = ModelIdentity(model_hash="aaa", model_family="gpt2", hidden_dim=64, num_layers=2)
    remote = ModelIdentity(model_hash="bbb", model_family="llama", hidden_dim=128, num_layers=4)

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        old_dir = registry._MAP_DIR
        try:
            registry._MAP_DIR = pathlib.Path(tmp)
            with caplog.at_level(logging.DEBUG, logger="avp.handshake"):
                CompatibilityResolver.resolve(local, remote)
        finally:
            registry._MAP_DIR = old_dir
    assert any("json_fallback" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end auto-negotiation integration tests
#
# These tests validate that the handshake can actually switch modes
# automatically using real (tiny) model objects, not just mock identities.
# They cover the full chain:  handshake → calibrate → save → re-discover.
# ---------------------------------------------------------------------------

if HAS_TORCH and HAS_TRANSFORMERS:
    from conftest import MockTokenizer


    class _CrossFamilyMockTokenizer(MockTokenizer):
        """Tokenizer with partial overlap: first half shared, second half unique."""

        def get_vocab(self):
            vocab = {}
            for i in range(self.vocab_size):
                if i < self.vocab_size // 2:
                    vocab[f"token_{i}"] = i
                else:
                    vocab[f"alt_token_{i}"] = i
            return vocab


    class _SharedMockTokenizer(MockTokenizer):
        """Tokenizer with full vocabulary overlap (same token names)."""

        def get_vocab(self):
            return {f"token_{i}": i for i in range(self.vocab_size)}


@requires_torch
@requires_transformers
class TestAutoNegotiationEndToEnd:
    """Integration tests: handshake auto-discovers communication mode
    using real model objects and tokenizers.
    """

    @pytest.fixture
    def gpt2_model(self):
        """Tiny GPT2 model (family='gpt2', dim=64)."""
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(
            vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128,
        )
        model = GPT2LMHeadModel(config)
        model.eval()
        return model

    @pytest.fixture
    def llama_model(self):
        """Tiny Llama model (family='llama', dim=128)."""
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config)
        model.eval()
        return model

    def test_cross_family_auto_discovers_vocab_overlap(
        self, gpt2_model, llama_model
    ):
        """Two models from different families with partial vocab overlap →
        handshake resolves LATENT via vocab_overlap when tokenizers provided.
        """
        import avp.rosetta.registry as registry

        src_tok = _SharedMockTokenizer(vocab_size=256)
        tgt_tok = _CrossFamilyMockTokenizer(vocab_size=256)

        src_identity = extract_model_identity(gpt2_model, tokenizer=src_tok)
        tgt_identity = extract_model_identity(llama_model, tokenizer=tgt_tok)

        # Sanity: different family, different dims → rules 1-2 miss
        assert src_identity.model_family != tgt_identity.model_family
        assert src_identity.hidden_dim != tgt_identity.hidden_dim
        # Different tokenizer hash → rule 3 misses
        assert src_identity.tokenizer_hash != tgt_identity.tokenizer_hash

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib
            old_dir = registry._MAP_DIR
            try:
                registry._MAP_DIR = pathlib.Path(tmp)
                result = CompatibilityResolver.resolve(
                    src_identity, tgt_identity,
                    source_tokenizer=src_tok,
                    target_tokenizer=tgt_tok,
                )
                assert result.mode == CommunicationMode.LATENT
                assert result.avp_map_id.startswith("vocab_overlap:")
            finally:
                registry._MAP_DIR = old_dir

    def test_cross_family_without_tokenizers_falls_to_json(
        self, gpt2_model, llama_model
    ):
        """Same model pair but no tokenizers passed → JSON.

        This is the scenario that actually happens today: transport.py
        calls resolve() without tokenizer objects.
        """
        import avp.rosetta.registry as registry

        src_tok = _SharedMockTokenizer(vocab_size=256)
        tgt_tok = _CrossFamilyMockTokenizer(vocab_size=256)

        src_identity = extract_model_identity(gpt2_model, tokenizer=src_tok)
        tgt_identity = extract_model_identity(llama_model, tokenizer=tgt_tok)

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib
            old_dir = registry._MAP_DIR
            try:
                registry._MAP_DIR = pathlib.Path(tmp)
                # Without tokenizers — simulates current transport.py behavior
                result = CompatibilityResolver.resolve(src_identity, tgt_identity)
                assert result.mode == CommunicationMode.JSON
            finally:
                registry._MAP_DIR = old_dir

    def test_calibrate_saves_then_handshake_rediscovers(
        self, gpt2_model, llama_model
    ):
        """Full chain: calibrate() auto-saves → subsequent resolve()
        without tokenizers finds the .avp-map file via rule 4.

        This is the key lifecycle test:
          1. First encounter (with tokenizers or explicit calibrate) → LATENT
          2. calibrate() auto-saves AVPMap to disk
          3. Second encounter (without tokenizers) → still LATENT via map file
        """
        import avp.rosetta.registry as registry
        from avp.rosetta.calibrate import calibrate

        src_tok = _SharedMockTokenizer(vocab_size=256)
        tgt_tok = _CrossFamilyMockTokenizer(vocab_size=256)

        src_identity = extract_model_identity(gpt2_model, tokenizer=src_tok)
        tgt_identity = extract_model_identity(llama_model, tokenizer=tgt_tok)

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib
            old_dir = registry._MAP_DIR
            try:
                registry._MAP_DIR = pathlib.Path(tmp)

                # Step 1: Without tokenizers or map file → JSON
                result = CompatibilityResolver.resolve(src_identity, tgt_identity)
                assert result.mode == CommunicationMode.JSON

                # Step 2: calibrate() with auto_save=True → writes .avp-map
                calibrate(
                    gpt2_model, llama_model, src_tok, tgt_tok,
                    device="cpu", auto_save=True,
                )

                # Step 3: Same resolve() WITHOUT tokenizers → LATENT via map file
                result = CompatibilityResolver.resolve(src_identity, tgt_identity)
                assert result.mode == CommunicationMode.LATENT
                assert result.avp_map_id  # non-empty map id
            finally:
                registry._MAP_DIR = old_dir

    def test_same_family_different_size_auto_discovers_vocab_mediated(
        self, gpt2_model,
    ):
        """Two GPT2 models (same family, different dims, shared tokenizer) →
        handshake resolves LATENT via shared_tokenizer (rule 3).

        This path works today without passing tokenizer objects, because
        it relies on tokenizer_hash in the identity.
        """
        from transformers import GPT2Config, GPT2LMHeadModel
        import avp.rosetta.registry as registry

        config_big = GPT2Config(
            vocab_size=256, n_embd=128, n_head=4, n_layer=3, n_positions=128,
        )
        model_small = gpt2_model
        model_big = GPT2LMHeadModel(config_big)
        model_big.eval()

        # Same tokenizer class → same tokenizer_hash
        tok = _SharedMockTokenizer(vocab_size=256)

        src_identity = extract_model_identity(model_small, tokenizer=tok)
        tgt_identity = extract_model_identity(model_big, tokenizer=tok)

        # Same tokenizer hash
        assert src_identity.tokenizer_hash == tgt_identity.tokenizer_hash
        # But different structure
        assert src_identity.hidden_dim != tgt_identity.hidden_dim

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib
            old_dir = registry._MAP_DIR
            try:
                registry._MAP_DIR = pathlib.Path(tmp)
                # No tokenizer objects needed — rule 3 uses hash from identity
                result = CompatibilityResolver.resolve(src_identity, tgt_identity)
                assert result.mode == CommunicationMode.LATENT
                assert result.avp_map_id.startswith("vocab:")
            finally:
                registry._MAP_DIR = old_dir

    def test_handshake_overlap_then_calibrate_then_project(
        self, llama_model
    ):
        """Full round-trip: handshake detects overlap → calibrate →
        project hidden state → result has correct target dimension.

        Validates the chain is usable, not just that resolve() returns LATENT.
        """
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        import avp.rosetta.registry as registry
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.project import vocab_overlap_projection

        # Second Llama model with different size
        config2 = LlamaConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, tie_word_embeddings=False,
        )
        llama_model_small = LlamaForCausalLM(config2)
        llama_model_small.eval()

        src_tok = _SharedMockTokenizer(vocab_size=256)
        tgt_tok = _CrossFamilyMockTokenizer(vocab_size=256)

        src_identity = extract_model_identity(llama_model, tokenizer=src_tok)
        tgt_identity = extract_model_identity(llama_model_small, tokenizer=tgt_tok)

        src_dim = llama_model.config.hidden_size
        tgt_dim = llama_model_small.config.hidden_size

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib
            old_dir = registry._MAP_DIR
            try:
                registry._MAP_DIR = pathlib.Path(tmp)

                # Handshake with tokenizers → vocab_overlap
                result = CompatibilityResolver.resolve(
                    src_identity, tgt_identity,
                    source_tokenizer=src_tok,
                    target_tokenizer=tgt_tok,
                )
                assert result.mode == CommunicationMode.LATENT
                assert result.avp_map_id.startswith("vocab_overlap:")

                # Calibrate (the step the caller must do after handshake)
                avp_map = calibrate(
                    llama_model, llama_model_small, src_tok, tgt_tok,
                    device="cpu",
                )

                # Project a dummy hidden state through the map
                source_lm_head = llama_model.get_output_embeddings()
                hidden = torch.randn(1, src_dim)
                projected = vocab_overlap_projection(
                    hidden,
                    source_lm_head_weight=source_lm_head.weight,
                    shared_target_embed_weight=avp_map.w_map,
                    src_indices=avp_map.src_indices,
                )
                assert projected.shape == (1, tgt_dim)
            finally:
                registry._MAP_DIR = old_dir

    def test_map_file_from_prior_calibrate_skips_vocab_overlap_compute(
        self, gpt2_model, llama_model
    ):
        """When a .avp-map file already exists, resolve() returns LATENT
        via rule 4 without needing tokenizers or computing overlap.

        This validates the "subsequent encounter" fast path.
        """
        import avp.rosetta.registry as registry
        from avp.rosetta.calibrate import calibrate
        from avp.rosetta.registry import map_id

        src_tok = _SharedMockTokenizer(vocab_size=256)
        tgt_tok = _CrossFamilyMockTokenizer(vocab_size=256)

        src_identity = extract_model_identity(gpt2_model, tokenizer=src_tok)
        tgt_identity = extract_model_identity(llama_model, tokenizer=tgt_tok)

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib
            old_dir = registry._MAP_DIR
            try:
                registry._MAP_DIR = pathlib.Path(tmp)

                # Pre-calibrate to create map file
                calibrate(
                    gpt2_model, llama_model, src_tok, tgt_tok,
                    device="cpu", auto_save=True,
                )

                # Now resolve WITHOUT tokenizers — rule 4 should catch it
                result = CompatibilityResolver.resolve(src_identity, tgt_identity)
                assert result.mode == CommunicationMode.LATENT
                expected_id = map_id(
                    src_identity.model_hash, tgt_identity.model_hash,
                )
                assert result.avp_map_id == expected_id
            finally:
                registry._MAP_DIR = old_dir

