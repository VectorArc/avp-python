"""Easy API: avp.think() and avp.generate().

    import avp

    # With any connector (Ollama, llama.cpp, etc.)
    from avp import OllamaConnector
    conn = OllamaConnector.from_ollama("qwen2.5:7b")
    context = avp.think("Analyze this", model=conn)
    answer = avp.generate("Solve it", model=conn, context=context)

    # With a HuggingFace model name (auto-creates connector)
    context = avp.think("Analyze this", model="Qwen/Qwen2.5-7B-Instruct")
    answer = avp.generate("Solve it", model="Qwen/Qwen2.5-7B-Instruct", context=context)

    # Or do both in one call
    answer = avp.generate("Analyze and solve: 24*17+3", model=conn)

For direct connector access (advanced):
    connector = avp.HuggingFaceConnector.from_pretrained("Qwen/...")
    context = connector.think("...", steps=20)
    answer = connector.generate("...", context=context)

Async variants (``avp.athink()`` / ``avp.agenerate()``) are planned
for a future release.
"""

import logging
import time as _time
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from .connectors.base import EngineConnector
    from .context import AVPContext
    from .context_store import ContextStore
    from .metrics import DebugConfig
    from .results import InspectResult

from .results import GenerateResult, ThinkResult
from .types import PayloadType

logger = logging.getLogger(__name__)

# Type alias for model specification: name (str) or pre-built connector.
ModelSpec = Union[str, "EngineConnector"]
"""A model specifier: HuggingFace model name (str) or EngineConnector instance."""


# ---------------------------------------------------------------------------
# Connector caching
# ---------------------------------------------------------------------------

_connector_cache: Dict[str, Any] = {}
_connector_lock = threading.Lock()


def _get_or_create_connector(model_name: str) -> Any:
    """Get or create a cached HuggingFaceConnector.

    Called when ``model=`` receives a string.  The connector is cached
    by name so repeated calls reuse the same instance.  Always creates
    a HuggingFaceConnector (requires ``avp[hf]``).
    """
    with _connector_lock:
        if model_name in _connector_cache:
            return _connector_cache[model_name]

    from .connectors.huggingface import HuggingFaceConnector

    connector = HuggingFaceConnector.from_pretrained(model_name)

    with _connector_lock:
        _connector_cache[model_name] = connector
    return connector


def clear_cache() -> None:
    """Free cached connectors to reclaim memory."""
    with _connector_lock:
        _connector_cache.clear()


def _resolve_model(model: Any) -> Any:
    """Resolve a model specifier to a connector instance.

    - ``str``: auto-creates and caches a HuggingFaceConnector (requires avp[hf]).
    - ``EngineConnector``: returned as-is (caller owns lifecycle).
    """
    if isinstance(model, str):
        return _get_or_create_connector(model)
    from .connectors.base import EngineConnector
    if isinstance(model, EngineConnector):
        return model
    from .errors import ConfigurationError
    raise ConfigurationError(
        f"model= must be a model name (str) or EngineConnector instance, "
        f"got {type(model).__name__}"
    )


def _resolve_model_label(model: Any) -> str:
    """Derive a human-readable model label for metrics and logging."""
    if model is None:
        return "unknown"
    if isinstance(model, str):
        return model
    try:
        identity = model.get_model_identity()
        if identity and identity.model_id:
            return identity.model_id
    except Exception:
        pass
    return type(model).__name__


# ---------------------------------------------------------------------------
# think()
# ---------------------------------------------------------------------------


def think(
    prompt: str,
    *,
    model: Optional[ModelSpec] = None,
    steps: int = 20,
    context: Optional["AVPContext"] = None,
    output: PayloadType = PayloadType.AUTO,
    collect_metrics: bool = False,
    debug_config: Optional["DebugConfig"] = None,
) -> ThinkResult:
    """Run latent thinking steps on a prompt.

    Returns a :class:`ThinkResult` that wraps an AVPContext.  Attribute
    access is delegated to the context, so ``result.past_key_values``
    works.  Tuple unpacking is supported for backward compatibility::

        context, metrics = avp.think(..., collect_metrics=True)

    Examples::

        # With a connector (any backend)
        from avp import OllamaConnector
        conn = OllamaConnector.from_ollama("qwen2.5:7b")
        result = avp.think("Analyze this", model=conn)

        # With a model name (auto-creates HuggingFace backend)
        result = avp.think("Analyze this", model="Qwen/Qwen2.5-7B-Instruct")

    Args:
        prompt: The text prompt to think about.
        model: A model name (str, auto-creates HuggingFaceConnector) or
            an :class:`EngineConnector` instance (any backend).
        steps: Number of latent thinking steps. Default 20.
        context: Prior AVPContext or ThinkResult to continue from.
        collect_metrics: If True, populate ``result.metrics``.
        debug_config: Enable debug diagnostics via ``DebugConfig``.
            Implies ``collect_metrics=True``.

    Returns:
        ThinkResult wrapping the AVPContext and optional metrics.
    """
    t_start = _time.perf_counter()

    from .errors import ConfigurationError

    if not isinstance(prompt, str):
        raise ConfigurationError(f"think() prompt must be str, got {type(prompt).__name__}")
    if not prompt:
        raise ConfigurationError("think() requires a non-empty prompt string")
    if model is None:
        raise ConfigurationError("think() requires model= (str or EngineConnector)")
    if not isinstance(output, PayloadType):
        raise ConfigurationError(
            f"think() output= must be a PayloadType value, got {type(output).__name__}. "
            "Use PayloadType.AUTO, PayloadType.KV_CACHE, or PayloadType.HIDDEN_STATE."
        )

    if debug_config is not None:
        collect_metrics = True

    # Unwrap ThinkResult if passed as context (backward compat)
    if isinstance(context, ThinkResult):
        context = context.context

    model_label = _resolve_model_label(model)
    resolved = _resolve_model(model)

    if not getattr(resolved, "can_think", False):
        raise ConfigurationError(
            f"{type(resolved).__name__} does not support latent thinking "
            f"(can_think=False). Use a connector with think() support "
            f"(HuggingFaceConnector, LlamaCppConnector, OllamaConnector)."
        )

    diagnostics = None
    if debug_config is not None:
        from .metrics import TransferDiagnostics
        diagnostics = TransferDiagnostics(target_model=model_label)

    t_think = _time.perf_counter()
    avp_context = resolved.think(
        prompt, steps=steps, context=context,
        output=output,
        _diagnostics=diagnostics,
    )
    think_duration = _time.perf_counter() - t_think

    if diagnostics is not None:
        logger.info("think() debug: %s", diagnostics.summary())

    logger.info(
        "think() model=%s steps=%d duration=%.3fs",
        model_label, steps, think_duration,
    )

    metrics = None
    if collect_metrics:
        from .metrics import ThinkMetrics

        metrics = ThinkMetrics(
            model=model_label,
            steps=steps,
            has_prior_context=context is not None,
            duration_s=_time.perf_counter() - t_start,
            think_duration_s=think_duration,
            diagnostics=diagnostics,
        )

    return ThinkResult(avp_context, metrics)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def generate(
    prompt: str = "",
    *,
    model: Optional[ModelSpec] = None,
    source_model: Optional[ModelSpec] = None,
    cross_model: bool = False,
    steps: int = 20,
    context: Optional["AVPContext"] = None,
    store: Optional["ContextStore"] = None,
    store_key: Optional[str] = None,
    prior_key: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    collect_metrics: bool = False,
    debug_config: Optional["DebugConfig"] = None,
    # Deprecated — use ``prompt`` instead
    content: Optional[str] = None,
    **kwargs: Any,
) -> GenerateResult:
    """Think about a prompt, optionally store/retrieve context, and generate text.

    With a connector::

        from avp import OllamaConnector
        conn = OllamaConnector.from_ollama("qwen2.5:7b")
        text = avp.generate("Solve: 2+2", model=conn)

    With a model name::

        text = avp.generate("Solve: 2+2", model="Qwen/Qwen2.5-7B-Instruct")

    Cross-model::

        text = avp.generate("Solve: 2+2", model=target_conn,
                            source_model=source_conn, cross_model=True)

    With context store for multi-turn::

        text = avp.generate("Solve: 2+2", model=conn, store=store,
                            store_key="agent-a")

    Args:
        prompt: The prompt text.
        model: A model name (str, auto-creates HuggingFaceConnector) or
            an :class:`EngineConnector` instance (any backend).
        source_model: Source model for cross-model projection.  Accepts
            the same types as ``model``.
        cross_model: Must be True to enable cross-model projection.
            Cross-model (Rosetta Stone) is experimental — accuracy varies
            by task type (structured tasks like math/code work well,
            comprehension tasks may degrade). Default False.
        steps: Number of latent thinking steps. Default 20. Set to 0 for
            text-only generation without latent context.
        context: Prior AVPContext to continue from.
        store: A ``ContextStore`` instance for automatic context management.
        store_key: Store the context under this key after thinking.
        prior_key: Retrieve prior context from store under this key.
        max_new_tokens: Max tokens for generation.
        temperature: Sampling temperature.
        collect_metrics: If True, populate ``result.metrics`` with
            a ``GenerateMetrics`` instance.
        debug_config: Enable debug diagnostics via ``DebugConfig``.
            Implies ``collect_metrics=True``.
        content: **Deprecated.** Use ``prompt`` instead. Will be removed in v2.0.

    Returns:
        GenerateResult (subclass of str).  Access metrics via
        ``result.metrics`` (None when ``collect_metrics=False``).
    """
    # Handle deprecated content= parameter
    if content is not None:
        import warnings as _w
        _w.warn(
            "generate(content=...) is deprecated, use generate(prompt=...) instead. "
            "The 'content' parameter will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        if prompt:
            from .errors import ConfigurationError
            raise ConfigurationError("Cannot pass both 'prompt' and 'content' to generate()")
        prompt = content
    from .errors import ConfigurationError

    if not isinstance(prompt, str):
        raise ConfigurationError(f"generate() prompt must be str, got {type(prompt).__name__}")
    if not prompt:
        raise ConfigurationError("generate() requires a non-empty prompt string")
    if model is None:
        raise ConfigurationError("generate() requires model= (str or EngineConnector)")

    t_start = _time.perf_counter()
    if (store_key is not None or prior_key is not None) and store is None:
        raise ConfigurationError("store_key/prior_key require store= (pass a ContextStore)")

    if debug_config is not None:
        collect_metrics = True

    # Unwrap ThinkResult if passed as context (backward compat)
    if isinstance(context, ThinkResult):
        context = context.context

    model_label = _resolve_model_label(model)
    source_label = _resolve_model_label(source_model) if source_model is not None else None

    diagnostics = None
    if debug_config is not None:
        from .metrics import TransferDiagnostics
        diagnostics = TransferDiagnostics(
            target_model=model_label,
            source_model=source_label,
        )

    # Cross-model path: think on source, project to target
    if source_model is not None and not cross_model:
        import warnings as _warnings
        _warnings.warn(
            "Cross-model projection (Rosetta Stone) is experimental. "
            "Falling back to text-only generation on the target model. "
            "Pass cross_model=True to avp.generate() to enable cross-model "
            "latent transfer. Accuracy varies by task: structured tasks "
            "(math, code) work well, comprehension tasks may degrade. "
            "See docs/BENCHMARKS.md for details.",
            UserWarning,
            stacklevel=2,
        )
        source_model = None
        if diagnostics is not None:
            diagnostics.source_model = None

    if source_model is not None:
        resolved_source = _resolve_model(source_model)
        resolved_target = _resolve_model(model)

        # Validate source can think (only needed when no context provided)
        if steps > 0 and context is None and not getattr(resolved_source, "can_think", False):
            raise ConfigurationError(
                f"{type(resolved_source).__name__} does not support latent thinking "
                f"(can_think=False). Use a source that supports think(), "
                f"or provide context= from a prior think() call."
            )

        # Retrieve prior context from store
        if prior_key is not None and store is not None:
            prior = store.get(prior_key)
            if prior is not None:
                context = prior

        # Use caller-provided context, or think to create one
        t_think = _time.perf_counter()
        if context is not None:
            source_context = context
        elif steps > 0:
            source_context = resolved_source.think(
                prompt, steps=steps,
                _diagnostics=diagnostics,
            )
        else:
            source_context = None
        think_duration = _time.perf_counter() - t_think

        # Store context
        stored = False
        if store_key is not None and store is not None and source_context is not None:
            store.store(store_key, source_context)
            stored = True

        t_gen = _time.perf_counter()
        text = resolved_target.generate(
            prompt,
            context=source_context,
            source=resolved_source,
            cross_model=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            _diagnostics=diagnostics,
            **kwargs,
        )
        generate_duration = _time.perf_counter() - t_gen

        logger.info(
            "generate() cross-model %s→%s think=%.3fs generate=%.3fs total=%.3fs",
            source_label, model_label, think_duration, generate_duration,
            _time.perf_counter() - t_start,
        )

        # Compare mode
        if debug_config is not None and debug_config.compare:
            _run_compare(diagnostics, resolved_target, prompt, text,
                         max_new_tokens, temperature)

        if diagnostics is not None:
            logger.info("generate() debug: %s", diagnostics.summary())

        metrics = None
        if collect_metrics:
            from .metrics import GenerateMetrics

            metrics = GenerateMetrics(
                model=model_label,
                steps=steps,
                has_prior_context=context is not None,
                stored=stored,
                duration_s=_time.perf_counter() - t_start,
                think_duration_s=think_duration,
                generate_duration_s=generate_duration,
                diagnostics=diagnostics,
            )

        return GenerateResult(text, metrics=metrics)

    # Same-model path
    resolved = _resolve_model(model)

    # Validate can_think when latent steps are requested
    if steps > 0 and not getattr(resolved, "can_think", False):
        raise ConfigurationError(
            f"{type(resolved).__name__} does not support latent thinking "
            f"(can_think=False). Either pass steps=0 for text-only generation, "
            f"or use a connector that supports think() "
            f"(HuggingFaceConnector, LlamaCppConnector, OllamaConnector)."
        )

    # Retrieve prior context from store
    if prior_key is not None and store is not None:
        prior = store.get(prior_key)
        if prior is not None:
            context = prior

    # Think — call connector directly to share the same diagnostics object
    t_think = _time.perf_counter()
    if steps > 0:
        avp_context = resolved.think(
            prompt, steps=steps, context=context,
            _diagnostics=diagnostics,
        )
    else:
        avp_context = context
    think_duration = _time.perf_counter() - t_think

    # Store context
    stored = False
    if store_key is not None and store is not None and avp_context is not None:
        store.store(store_key, avp_context)
        stored = True

    # Generate text (connector already obtained above)
    t_gen = _time.perf_counter()
    text = resolved.generate(
        prompt,
        context=avp_context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        _diagnostics=diagnostics,
        **kwargs,
    )
    generate_duration = _time.perf_counter() - t_gen

    logger.debug(
        "generate() model=%s steps=%d stored=%s prior=%s",
        model_label, steps, store_key, prior_key,
    )
    logger.info(
        "generate() think=%.3fs generate=%.3fs total=%.3fs",
        think_duration, generate_duration, _time.perf_counter() - t_start,
    )

    # Compare mode
    if debug_config is not None and debug_config.compare:
        _run_compare(diagnostics, resolved, prompt, text, max_new_tokens, temperature)

    if diagnostics is not None:
        logger.info("generate() debug: %s", diagnostics.summary())

    metrics = None
    if collect_metrics:
        from .metrics import GenerateMetrics

        metrics = GenerateMetrics(
            model=model_label,
            steps=steps,
            has_prior_context=context is not None,
            stored=stored,
            duration_s=_time.perf_counter() - t_start,
            think_duration_s=think_duration,
            generate_duration_s=generate_duration,
            diagnostics=diagnostics,
        )

    return GenerateResult(text, metrics=metrics)


def _run_compare(
    diagnostics: Optional[Any],
    connector: Any,
    prompt: str,
    latent_output: str,
    max_new_tokens: int,
    temperature: float,
) -> None:
    """Generate text-only baseline and compare with latent output."""
    if diagnostics is None:
        return
    try:
        baseline = connector.generate(
            prompt,
            context=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        diagnostics.text_baseline_output = baseline
        diagnostics.text_overlap = _compute_word_overlap(latent_output, baseline)
    except Exception as exc:
        logger.warning("compare mode: text baseline generation failed: %s", exc)
        diagnostics.warnings.append(f"compare baseline failed: {exc}")


def _compute_word_overlap(a: str, b: str) -> float:
    """Return word overlap ratio between two strings (0.0 to 1.0)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


# ---------------------------------------------------------------------------
# inspect()
# ---------------------------------------------------------------------------


def inspect(data: bytes) -> "InspectResult":
    """Inspect an AVP binary payload without loading models.

    Decodes the header and metadata from raw AVP wire bytes. Useful for
    debugging wire payloads, logging, and observability tools.

    Args:
        data: Raw AVP binary bytes (from ``AVPContext.to_bytes()`` or wire).

    Returns:
        InspectResult with decoded header/metadata fields.

    Raises:
        DecodeError: If data is not valid AVP binary.
    """
    from .codec import decode as avp_decode
    from .results import InspectResult

    msg = avp_decode(data)
    h = msg.header
    m = msg.metadata

    return InspectResult(
        version=h.version,
        flags=h.flags,
        compressed=bool(h.flags & 0x01),
        has_map=bool(h.flags & 0x02),
        kv_cache=bool(h.flags & 0x04),
        payload_length=h.payload_length,
        metadata_length=h.metadata_length,
        model_id=m.model_id,
        hidden_dim=m.hidden_dim,
        num_layers=m.num_layers,
        payload_type=m.payload_type.name if hasattr(m.payload_type, "name") else str(m.payload_type),
        dtype=m.dtype.name if hasattr(m.dtype, "name") else str(m.dtype),
        tensor_shape=list(m.tensor_shape) if m.tensor_shape else [],
        mode=m.mode.name if hasattr(m.mode, "name") else str(m.mode),
        session_id=m.session_id,
        source_agent_id=m.source_agent_id,
        target_agent_id=m.target_agent_id,
        avp_map_id=m.avp_map_id,
        extra=dict(m.extra) if m.extra else {},
        raw_size=msg.raw_size,
    )
