"""Thread-safe store for AVPContext objects with TTL expiry.

Provides the ad-hoc ``_latent_store: dict[str, AVPContext]`` that every
framework integration reinvents, as a proper SDK utility with TTL cleanup
and thread safety.

Usage::

    import avp

    store = avp.ContextStore(default_ttl=300)
    ctx = avp.think("hello", model="Qwen/...")
    store.store("task-1", ctx)
    ctx = store.get("task-1")   # None after TTL expires
"""

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .context import AVPContext


@dataclass
class _Entry:
    """Internal wrapper around a stored AVPContext."""

    context: "AVPContext"
    stored_at: float
    ttl: float

    @property
    def is_expired(self) -> bool:
        return time.time() > self.stored_at + self.ttl


class ContextStore:
    """Thread-safe AVPContext store with per-entry TTL.

    Default TTL is 300 s (5 min) — shorter than SessionManager's 1 hr
    because tensor-backed contexts are memory-heavy.

    Args:
        default_ttl: Default time-to-live in seconds for stored entries.
    """

    def __init__(self, default_ttl: float = 300.0):
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl

    def store(self, key: str, context: "AVPContext", ttl: Optional[float] = None) -> None:
        """Store an AVPContext under *key*, replacing any existing entry.

        Args:
            key: Lookup key (e.g. task ID, session ID).
            context: The AVPContext to store.
            ttl: Per-entry TTL in seconds.  ``None`` uses the store default.
        """
        entry = _Entry(
            context=context,
            stored_at=time.time(),
            ttl=ttl if ttl is not None else self._default_ttl,
        )
        with self._lock:
            self._entries[key] = entry

    def get(self, key: str) -> Optional["AVPContext"]:
        """Retrieve an AVPContext by key, or ``None`` if missing/expired.

        Expired entries are lazily removed on access.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._entries[key]
                return None
            return entry.context

    def remove(self, key: str) -> bool:
        """Remove an entry.  Returns ``True`` if it existed."""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def keys(self) -> List[str]:
        """Return keys of all active (non-expired) entries."""
        now = time.time()
        with self._lock:
            return [
                k for k, e in self._entries.items()
                if now <= e.stored_at + e.ttl
            ]

    def cleanup_expired(self) -> int:
        """Remove all expired entries.  Returns count removed."""
        with self._lock:
            expired = [k for k, e in self._entries.items() if e.is_expired]
            for k in expired:
                del self._entries[k]
            return len(expired)

    @property
    def active_count(self) -> int:
        """Number of active (non-expired) entries."""
        now = time.time()
        with self._lock:
            return sum(1 for e in self._entries.values() if now <= e.stored_at + e.ttl)
