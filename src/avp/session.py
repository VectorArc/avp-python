"""AVP session management.

Sessions track negotiated communication state between agent pairs.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import CommunicationMode, ModelIdentity


@dataclass
class Session:
    """An active AVP communication session."""

    session_id: str = ""
    agents: List[str] = field(default_factory=list)
    identities: Dict[str, ModelIdentity] = field(default_factory=dict)
    mode: CommunicationMode = CommunicationMode.JSON
    created_at: float = 0.0
    ttl: float = 3600.0  # Default 1 hour

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl


class SessionManager:
    """Thread-safe session store with expiry management."""

    def __init__(self, default_ttl: float = 3600.0):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl

    def create(
        self,
        agents: List[str],
        identities: Dict[str, ModelIdentity],
        mode: CommunicationMode,
        ttl: Optional[float] = None,
    ) -> Session:
        """Create a new session.

        Args:
            agents: List of agent IDs participating.
            identities: Map of agent_id → ModelIdentity.
            mode: Negotiated communication mode.
            ttl: Session time-to-live in seconds.

        Returns:
            The newly created Session.
        """
        session = Session(
            session_id=uuid.uuid4().hex,
            agents=agents,
            identities=identities,
            mode=mode,
            created_at=time.time(),
            ttl=ttl if ttl is not None else self._default_ttl,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, returning None if expired or not found."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                return None
            return session

    def invalidate(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def find(self, agent_id: str) -> List[Session]:
        """Find all active sessions involving an agent."""
        with self._lock:
            now = time.time()
            return [
                s for s in self._sessions.values()
                if agent_id in s.agents and not s.is_expired
            ]

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count of removed sessions."""
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if s.is_expired
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    @property
    def active_count(self) -> int:
        """Number of active (non-expired) sessions."""
        with self._lock:
            return sum(1 for s in self._sessions.values() if not s.is_expired)
