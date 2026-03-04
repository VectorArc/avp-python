"""Tests for AVP ContextStore."""

import threading
import time

from avp.context import AVPContext
from avp.context_store import ContextStore


def _make_ctx(content: str = "test") -> AVPContext:
    """Create a minimal AVPContext for testing."""
    return AVPContext(
        past_key_values=None,
        model_hash=f"hash-{content}",
        num_steps=0,
        seq_len=0,
    )


# --- Basic operations ---


def test_store_and_get():
    store = ContextStore()
    ctx = _make_ctx("hello")
    store.store("k1", ctx)
    assert store.get("k1") is ctx


def test_get_nonexistent():
    store = ContextStore()
    assert store.get("missing") is None


def test_overwrite():
    store = ContextStore()
    ctx1 = _make_ctx("first")
    ctx2 = _make_ctx("second")
    store.store("k1", ctx1)
    store.store("k1", ctx2)
    assert store.get("k1") is ctx2


def test_remove_existing():
    store = ContextStore()
    store.store("k1", _make_ctx())
    assert store.remove("k1") is True
    assert store.get("k1") is None


def test_remove_nonexistent():
    store = ContextStore()
    assert store.remove("nope") is False


def test_clear():
    store = ContextStore()
    store.store("a", _make_ctx("1"))
    store.store("b", _make_ctx("2"))
    store.clear()
    assert store.get("a") is None
    assert store.get("b") is None
    assert store.active_count == 0


# --- TTL ---


def test_expired_entry_returns_none():
    store = ContextStore(default_ttl=0.01)
    store.store("k1", _make_ctx())
    time.sleep(0.02)
    assert store.get("k1") is None


def test_per_entry_ttl():
    store = ContextStore(default_ttl=3600)
    store.store("short", _make_ctx("x"), ttl=0.01)
    store.store("long", _make_ctx("y"), ttl=3600)
    time.sleep(0.02)
    assert store.get("short") is None
    assert store.get("long") is not None


def test_cleanup_expired():
    store = ContextStore()
    store.store("expire", _make_ctx("x"), ttl=0.01)
    store.store("keep", _make_ctx("y"), ttl=3600)
    time.sleep(0.02)
    removed = store.cleanup_expired()
    assert removed == 1
    assert store.active_count == 1


# --- Keys / counts ---


def test_keys_active_only():
    store = ContextStore()
    store.store("a", _make_ctx("1"), ttl=3600)
    store.store("b", _make_ctx("2"), ttl=0.01)
    time.sleep(0.02)
    keys = store.keys()
    assert "a" in keys
    assert "b" not in keys


def test_active_count():
    store = ContextStore()
    assert store.active_count == 0
    store.store("a", _make_ctx("1"))
    store.store("b", _make_ctx("2"))
    assert store.active_count == 2


def test_active_count_excludes_expired():
    store = ContextStore()
    store.store("a", _make_ctx("1"), ttl=3600)
    store.store("b", _make_ctx("2"), ttl=0.01)
    time.sleep(0.02)
    assert store.active_count == 1


# --- Thread safety ---


def test_concurrent_operations():
    store = ContextStore()
    errors = []

    def writer(tid: int):
        try:
            for i in range(100):
                store.store(f"t{tid}-{i}", _make_ctx(f"{tid}-{i}"))
        except Exception as e:
            errors.append(e)

    def reader(tid: int):
        try:
            for i in range(100):
                store.get(f"t{tid}-{i}")
        except Exception as e:
            errors.append(e)

    threads = []
    for t in range(4):
        threads.append(threading.Thread(target=writer, args=(t,)))
        threads.append(threading.Thread(target=reader, args=(t,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# --- Edge cases ---


def test_empty_key():
    store = ContextStore()
    ctx = _make_ctx()
    store.store("", ctx)
    assert store.get("") is ctx


def test_context_with_metadata():
    store = ContextStore()
    ctx = AVPContext(
        past_key_values=None,
        model_hash="abc",
        num_steps=10,
        seq_len=100,
        model_family="qwen2",
        hidden_dim=4096,
        num_layers=32,
    )
    store.store("k1", ctx)
    retrieved = store.get("k1")
    assert retrieved is not None
    assert retrieved.model_hash == "abc"
    assert retrieved.model_family == "qwen2"
    assert retrieved.hidden_dim == 4096
