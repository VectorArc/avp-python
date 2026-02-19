"""AVP JSON fallback for cross-model communication.

When models are incompatible for latent communication, agents fall back
to structured JSON messages.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .types import AVP_VERSION_STRING


@dataclass
class JSONMessage:
    """A JSON fallback message for cross-model communication."""

    avp_version: str = AVP_VERSION_STRING
    session_id: str = ""
    source_agent_id: str = ""
    target_agent_id: str = ""
    content: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "avp_version": self.avp_version,
            "session_id": self.session_id,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "content": self.content,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JSONMessage":
        return cls(
            avp_version=d.get("avp_version", AVP_VERSION_STRING),
            session_id=d.get("session_id", ""),
            source_agent_id=d.get("source_agent_id", ""),
            target_agent_id=d.get("target_agent_id", ""),
            content=d.get("content", ""),
            extra=d.get("extra", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> "JSONMessage":
        return cls.from_dict(json.loads(data))


@dataclass
class FallbackRequest:
    """Request to switch from latent to JSON fallback mode."""

    session_id: str = ""
    reason: str = ""
    perplexity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reason": self.reason,
            "perplexity_score": self.perplexity_score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FallbackRequest":
        return cls(
            session_id=d.get("session_id", ""),
            reason=d.get("reason", ""),
            perplexity_score=d.get("perplexity_score", 0.0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> "FallbackRequest":
        return cls.from_dict(json.loads(data))
