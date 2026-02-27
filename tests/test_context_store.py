"""Tests for AVP ContextStore."""

import threading
import time

from avp.context_store import ContextStore
from avp.easy import PackedMessage


# --- Basic operations ---


def test_store_and_get():
    store = ContextStore()
    msg = PackedMessage(content="hello")
    store.store("k1", msg)
    assert store.get("k1") is msg


def test_get_nonexistent():
    store = ContextStore()
    assert store.get("missing") is None


def test_overwrite():
    store = ContextStore()
    msg1 = PackedMessage(content="first")
    msg2 = PackedMessage(content="second")
    store.store("k1", msg1)
    store.store("k1", msg2)
    assert store.get("k1") is msg2


def test_remove_existing():
    store = ContextStore()
    store.store("k1", PackedMessage(content="x"))
    assert store.remove("k1") is True
    assert store.get("k1") is None


def test_remove_nonexistent():
    store = ContextStore()
    assert store.remove("nope") is False


def test_clear():
    store = ContextStore()
    store.store("a", PackedMessage(content="1"))
    store.store("b", PackedMessage(content="2"))
    store.clear()
    assert store.get("a") is None
    assert store.get("b") is None
    assert store.active_count == 0


# --- TTL ---


def test_expired_entry_returns_none():
    store = ContextStore(default_ttl=0.01)
    store.store("k1", PackedMessage(content="x"))
    time.sleep(0.02)
    assert store.get("k1") is None


def test_per_entry_ttl():
    store = ContextStore(default_ttl=3600)
    store.store("short", PackedMessage(content="x"), ttl=0.01)
    store.store("long", PackedMessage(content="y"), ttl=3600)
    time.sleep(0.02)
    assert store.get("short") is None
    assert store.get("long") is not None


def test_cleanup_expired():
    store = ContextStore()
    store.store("expire", PackedMessage(content="x"), ttl=0.01)
    store.store("keep", PackedMessage(content="y"), ttl=3600)
    time.sleep(0.02)
    removed = store.cleanup_expired()
    assert removed == 1
    assert store.active_count == 1


# --- Keys / counts ---


def test_keys_active_only():
    store = ContextStore()
    store.store("a", PackedMessage(content="1"), ttl=3600)
    store.store("b", PackedMessage(content="2"), ttl=0.01)
    time.sleep(0.02)
    keys = store.keys()
    assert "a" in keys
    assert "b" not in keys


def test_active_count():
    store = ContextStore()
    assert store.active_count == 0
    store.store("a", PackedMessage(content="1"))
    store.store("b", PackedMessage(content="2"))
    assert store.active_count == 2


def test_active_count_excludes_expired():
    store = ContextStore()
    store.store("a", PackedMessage(content="1"), ttl=3600)
    store.store("b", PackedMessage(content="2"), ttl=0.01)
    time.sleep(0.02)
    assert store.active_count == 1


# --- Thread safety ---


def test_concurrent_operations():
    store = ContextStore()
    errors = []

    def writer(tid: int):
        try:
            for i in range(100):
                store.store(f"t{tid}-{i}", PackedMessage(content=f"{tid}-{i}"))
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
    msg = PackedMessage(content="x")
    store.store("", msg)
    assert store.get("") is msg


def test_packed_message_with_identity():
    store = ContextStore()
    msg = PackedMessage(content="hi", identity={"model_id": "test"}, model="test-model")
    store.store("k1", msg)
    retrieved = store.get("k1")
    assert retrieved is not None
    assert retrieved.identity == {"model_id": "test"}
    assert retrieved.model == "test-model"
