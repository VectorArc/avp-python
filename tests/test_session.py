"""Tests for AVP session management."""

import time

import pytest

from avp.session import Session, SessionManager
from avp.types import CommunicationMode, ModelIdentity


# --- Session ---


def test_session_not_expired():
    s = Session(session_id="s1", created_at=time.time(), ttl=3600)
    assert not s.is_expired


def test_session_expired():
    s = Session(session_id="s1", created_at=time.time() - 10, ttl=5)
    assert s.is_expired


# --- SessionManager ---


def test_create_session():
    mgr = SessionManager()
    s = mgr.create(
        agents=["a1", "a2"],
        identities={"a1": ModelIdentity(model_family="llama")},
        mode=CommunicationMode.LATENT,
    )
    assert len(s.session_id) > 0
    assert s.agents == ["a1", "a2"]
    assert s.mode == CommunicationMode.LATENT
    assert not s.is_expired


def test_get_session():
    mgr = SessionManager()
    s = mgr.create(
        agents=["a1"],
        identities={},
        mode=CommunicationMode.JSON,
    )
    retrieved = mgr.get(s.session_id)
    assert retrieved is not None
    assert retrieved.session_id == s.session_id


def test_get_nonexistent_session():
    mgr = SessionManager()
    assert mgr.get("nonexistent") is None


def test_get_expired_session():
    mgr = SessionManager()
    s = mgr.create(
        agents=["a1"],
        identities={},
        mode=CommunicationMode.JSON,
        ttl=0.01,
    )
    time.sleep(0.02)
    assert mgr.get(s.session_id) is None


def test_invalidate_session():
    mgr = SessionManager()
    s = mgr.create(agents=["a1"], identities={}, mode=CommunicationMode.JSON)
    assert mgr.invalidate(s.session_id) is True
    assert mgr.get(s.session_id) is None
    assert mgr.invalidate(s.session_id) is False


def test_find_sessions_by_agent():
    mgr = SessionManager()
    mgr.create(agents=["a1", "a2"], identities={}, mode=CommunicationMode.LATENT)
    mgr.create(agents=["a1", "a3"], identities={}, mode=CommunicationMode.JSON)
    mgr.create(agents=["a2", "a3"], identities={}, mode=CommunicationMode.JSON)

    found = mgr.find("a1")
    assert len(found) == 2

    found = mgr.find("a3")
    assert len(found) == 2

    found = mgr.find("nonexistent")
    assert len(found) == 0


def test_cleanup_expired():
    mgr = SessionManager()
    mgr.create(agents=["a1"], identities={}, mode=CommunicationMode.JSON, ttl=0.01)
    mgr.create(agents=["a2"], identities={}, mode=CommunicationMode.JSON, ttl=3600)
    time.sleep(0.02)

    removed = mgr.cleanup_expired()
    assert removed == 1
    assert mgr.active_count == 1


def test_active_count():
    mgr = SessionManager()
    assert mgr.active_count == 0
    mgr.create(agents=["a1"], identities={}, mode=CommunicationMode.JSON)
    mgr.create(agents=["a2"], identities={}, mode=CommunicationMode.JSON)
    assert mgr.active_count == 2
