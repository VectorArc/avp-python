"""Result types for the AVP Easy API.

``ThinkResult`` and ``GenerateResult`` provide a stable return type that
can grow new fields without changing the function signature.

    result = avp.think("prompt", model="...")
    result.context          # AVPContext
    result.metrics          # ThinkMetrics or None
    result.past_key_values  # delegates to context

    result = avp.generate("prompt", model="...")
    print(result)           # works — GenerateResult IS a str
    result.metrics          # GenerateMetrics or None
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

if TYPE_CHECKING:
    from .context import AVPContext
    from .metrics import GenerateMetrics, ThinkMetrics


@dataclass
class InspectResult:
    """Result of ``avp.inspect()`` — decoded AVP wire payload metadata."""

    version: int = 0
    flags: int = 0
    compressed: bool = False
    has_map: bool = False
    kv_cache: bool = False
    payload_length: int = 0
    metadata_length: int = 0
    model_id: str = ""
    hidden_dim: int = 0
    num_layers: int = 0
    payload_type: str = ""
    dtype: str = ""
    tensor_shape: List[int] = field(default_factory=list)
    mode: str = ""
    session_id: str = ""
    source_agent_id: str = ""
    target_agent_id: str = ""
    avp_map_id: str = ""
    extra: Dict[str, str] = field(default_factory=dict)
    raw_size: int = 0


class ThinkResult:
    """Result of ``avp.think()``.

    Wraps an :class:`AVPContext` with optional metrics.  Attribute access
    is delegated to the underlying context, so existing code like
    ``result.past_key_values`` continues to work.

    Supports tuple unpacking for backward compatibility::

        context, metrics = avp.think(..., collect_metrics=True)
    """

    __slots__ = ("context", "metrics")

    def __init__(
        self, context: "AVPContext", metrics: Optional["ThinkMetrics"] = None
    ) -> None:
        self.context: "AVPContext" = context
        self.metrics: Optional["ThinkMetrics"] = metrics

    # --- Backward compat: delegate attribute access to context ---

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found on ThinkResult itself.
        # Guard against infinite recursion if context is not set
        # (e.g., during unpickling before __init__ runs).
        if name == "context":
            raise AttributeError("ThinkResult has no context (not initialized)")
        return getattr(self.context, name)

    # --- Backward compat: tuple unpacking ---

    def __iter__(self) -> Iterator:
        """Support ``context, metrics = avp.think(...)``."""
        return iter((self.context, self.metrics))

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return (
            f"ThinkResult(context={self.context!r}, "
            f"metrics={'...' if self.metrics else None})"
        )


class GenerateResult(str):
    """Result of ``avp.generate()``.

    Subclasses :class:`str` so all string operations work transparently.
    Access metrics via the ``.metrics`` attribute::

        result = avp.generate("prompt", model="...")
        print(result)           # prints the generated text
        len(result)             # string length
        result.metrics          # GenerateMetrics or None
    """

    metrics: Optional["GenerateMetrics"]

    def __new__(
        cls,
        text: str = "",
        metrics: Optional["GenerateMetrics"] = None,
    ) -> "GenerateResult":
        instance = super().__new__(cls, text)
        instance.metrics = metrics  # type: ignore[attr-defined]
        return instance

    def __repr__(self) -> str:
        text_preview = str(self)[:80]
        if len(str(self)) > 80:
            text_preview += "..."
        return (
            f"GenerateResult({text_preview!r}, "
            f"metrics={'...' if self.metrics else None})"
        )

    def __reduce__(self) -> tuple:
        """Support pickling — preserves metrics across serialize/deserialize."""
        return (GenerateResult, (str(self), self.metrics))
