"""Tests for AVP handshake negotiation."""

import pytest

from avp.handshake import (
    CompatibilityResolver,
    HelloMessage,
    compute_model_hash,
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
