"""Tests for AVP handshake negotiation."""

import tempfile

import pytest

from avp.handshake import (
    CompatibilityResolver,
    HelloMessage,
    compute_model_hash,
    compute_tokenizer_hash,
    extract_model_identity,
)
from avp.types import CommunicationMode, ModelIdentity


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
    assert result.avp_map_id == f"vocab:abcdef1234567890"


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
