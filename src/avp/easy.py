"""Easy API: avp.think() and avp.generate().

    import avp

    # Get latent context
    context = avp.think("Analyze this problem", model="Qwen/Qwen2.5-7B-Instruct")

    # Generate with latent context
    answer = avp.generate("Solve it", model="Qwen/Qwen2.5-7B-Instruct", context=context)

    # Or do both in one call
    answer = avp.generate("Analyze and solve: 24*17+3", model="Qwen/Qwen2.5-7B-Instruct")

For direct connector access (advanced):
    connector = avp.HuggingFaceConnector.from_pretrained("Qwen/...")
    context = connector.think("...", steps=20)
    answer = connector.generate("...", context=context)
"""

import logging
import time as _time
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

if TYPE_CHECKING:
    from .context import AVPContext
    from .context_store import ContextStore
    from .metrics import DebugConfig, GenerateMetrics, ThinkMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

_identity_cache: Dict[str, Dict[str, Any]] = {}
_identity_lock = threading.Lock()


def _get_local_identity(model_name: str) -> Optional[Dict[str, Any]]:
    """Extract model identity from HuggingFace config (downloads config only, not weights).

    Returns None if transformers is not installed.
    """
    with _identity_lock:
        if model_name in _identity_cache:
            return _identity_cache[model_name]

    try:
        from transformers import AutoConfig
        from .handshake import compute_model_hash
    except ImportError:
        return {"model_id": model_name}

    try:
        config = AutoConfig.from_pretrained(model_name)
        cfg = config.to_dict()
        hidden_dim = cfg.get("hidden_size", 0)
        num_layers = cfg.get("num_hidden_layers", 0)
        model_family = cfg.get("model_type", "")
        model_hash = compute_model_hash(cfg)

        identity = {
            "model_hash": model_hash,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "model_family": model_family,
            "model_id": model_name,
        }
    except Exception:
        logger.warning("Failed to extract identity for %s", model_name, exc_info=True)
        identity = {"model_id": model_name}

    with _identity_lock:
        _identity_cache[model_name] = identity
    return identity


# ---------------------------------------------------------------------------
# Connector caching
# ---------------------------------------------------------------------------

_connector_cache: Dict[str, Any] = {}
_connector_lock = threading.Lock()


def _get_or_create_connector(model_name: str) -> Any:
    """Get or create a cached HuggingFaceConnector for latent operations."""
    with _connector_lock:
        if model_name in _connector_cache:
            return _connector_cache[model_name]

    from .connectors.huggingface import HuggingFaceConnector

    connector = HuggingFaceConnector.from_pretrained(model_name)

    with _connector_lock:
        _connector_cache[model_name] = connector
    return connector


def clear_cache() -> None:
    """Free cached connectors and identities to reclaim memory."""
    with _connector_lock:
        _connector_cache.clear()
    with _identity_lock:
        _identity_cache.clear()


# ---------------------------------------------------------------------------
# think()
# ---------------------------------------------------------------------------


def think(
    prompt: str,
    *,
    model: str,
    steps: int = 20,
    context: Optional["AVPContext"] = None,
    collect_metrics: bool = False,
    debug_config: Optional["DebugConfig"] = None,
) -> Union["AVPContext", Tuple["AVPContext", "ThinkMetrics"]]:
    """Run latent thinking steps on a prompt. Returns AVPContext.

    Equivalent to creating a HuggingFaceConnector and calling
    ``connector.think()``, but handles model loading and caching for you.

    Examples::

        context = avp.think("Analyze this", model="Qwen/Qwen2.5-7B-Instruct")
        context = avp.think("Continue", model="Qwen/...", context=prior_context)

    Args:
        prompt: The text prompt to think about.
        model: HuggingFace model name/path (required).
        steps: Number of latent thinking steps. Default 20.
        context: Prior AVPContext to continue from.
        collect_metrics: If True, return ``(AVPContext, ThinkMetrics)`` tuple.
        debug_config: Enable debug diagnostics via ``DebugConfig``.
            Implies ``collect_metrics=True``.

    Returns:
        AVPContext containing the KV-cache and metadata.
        If collect_metrics=True or debug_config is set, returns (AVPContext, ThinkMetrics).
    """
    t_start = _time.perf_counter()

    if not isinstance(prompt, str):
        from .errors import ConfigurationError
        raise ConfigurationError(f"think() prompt must be str, got {type(prompt).__name__}")

    if debug_config is not None:
        collect_metrics = True

    diagnostics = None
    if debug_config is not None:
        from .metrics import TransferDiagnostics
        diagnostics = TransferDiagnostics(target_model=model)

    connector = _get_or_create_connector(model)
    t_think = _time.perf_counter()
    avp_context = connector.think(
        prompt, steps=steps, context=context,
        _diagnostics=diagnostics,
    )
    think_duration = _time.perf_counter() - t_think

    if diagnostics is not None:
        logger.info("think() debug: %s", diagnostics.summary())

    logger.info(
        "think() model=%s steps=%d duration=%.3fs",
        model, steps, think_duration,
    )

    if collect_metrics:
        from .metrics import ThinkMetrics

        metrics = ThinkMetrics(
            model=model,
            steps=steps,
            has_prior_context=context is not None,
            duration_s=_time.perf_counter() - t_start,
            think_duration_s=think_duration,
            diagnostics=diagnostics,
        )
        return avp_context, metrics

    return avp_context


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def generate(
    prompt: str = "",
    *,
    model: str,
    source_model: Optional[str] = None,
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
) -> Union[str, Tuple[str, "GenerateMetrics"]]:
    """Think about a prompt, optionally store/retrieve context, and generate text.

    One-liner for the common think + generate pattern::

        text = avp.generate("Solve: 2+2", model="Qwen/Qwen2.5-7B-Instruct")

    Cross-model (Rosetta projection, experimental)::

        text = avp.generate("Solve: 2+2", model="target",
                            source_model="source", cross_model=True)

    With context store for multi-turn::

        text = avp.generate("Solve: 2+2", model=M, store=store,
                            store_key="agent-a")

    Args:
        prompt: The prompt text.
        model: HuggingFace model name/path (required).
        source_model: Source model for cross-model projection. When set,
            thinks on the source model and projects to the target (``model``)
            via Rosetta Stone. Requires ``cross_model=True``.
        cross_model: Must be True to enable cross-model projection.
            Cross-model (Rosetta Stone) is experimental — accuracy varies
            by task type. Default False.
        steps: Number of latent thinking steps. Default 20. Set to 0 for
            text-only generation without latent context.
        context: Prior AVPContext to continue from.
        store: A ``ContextStore`` instance for automatic context management.
        store_key: Store the context under this key after thinking.
        prior_key: Retrieve prior context from store under this key.
        max_new_tokens: Max tokens for generation.
        temperature: Sampling temperature.
        collect_metrics: If True, return ``(str, GenerateMetrics)`` tuple.
        debug_config: Enable debug diagnostics via ``DebugConfig``.
            Implies ``collect_metrics=True``.
        content: **Deprecated.** Use ``prompt`` instead. Will be removed in v2.0.

    Returns:
        Generated text response.
        If collect_metrics=True or debug_config is set, returns (str, GenerateMetrics).
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
    if not prompt:
        from .errors import ConfigurationError
        raise ConfigurationError("generate() requires a prompt string")

    t_start = _time.perf_counter()

    if not isinstance(prompt, str):
        from .errors import ConfigurationError
        raise ConfigurationError(f"generate() prompt must be str, got {type(prompt).__name__}")
    if (store_key is not None or prior_key is not None) and store is None:
        from .errors import ConfigurationError
        raise ConfigurationError("store_key/prior_key require store= (pass a ContextStore)")

    if debug_config is not None:
        collect_metrics = True

    diagnostics = None
    if debug_config is not None:
        from .metrics import TransferDiagnostics
        diagnostics = TransferDiagnostics(
            target_model=model,
            source_model=source_model,
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
        source_model = None  # fall through to same-model path on target
        if diagnostics is not None:
            diagnostics.source_model = None

    if source_model is not None:
        source_connector = _get_or_create_connector(source_model)
        target_connector = _get_or_create_connector(model)

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
            source_context = source_connector.think(
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
        text = target_connector.generate(
            prompt,
            context=source_context,
            source=source_connector,
            cross_model=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            _diagnostics=diagnostics,
        )
        generate_duration = _time.perf_counter() - t_gen

        logger.info(
            "generate() cross-model %s→%s think=%.3fs generate=%.3fs total=%.3fs",
            source_model, model, think_duration, generate_duration,
            _time.perf_counter() - t_start,
        )

        # Compare mode
        if debug_config is not None and debug_config.compare:
            _run_compare(diagnostics, target_connector, prompt, text,
                         max_new_tokens, temperature)

        if diagnostics is not None:
            logger.info("generate() debug: %s", diagnostics.summary())

        if collect_metrics:
            from .metrics import GenerateMetrics

            metrics = GenerateMetrics(
                model=model,
                steps=steps,
                has_prior_context=context is not None,
                stored=stored,
                duration_s=_time.perf_counter() - t_start,
                think_duration_s=think_duration,
                generate_duration_s=generate_duration,
                diagnostics=diagnostics,
            )
            return text, metrics

        return text

    # Same-model path

    # Retrieve prior context from store
    if prior_key is not None and store is not None:
        prior = store.get(prior_key)
        if prior is not None:
            context = prior

    # Think — call connector directly to share the same diagnostics object
    connector = _get_or_create_connector(model)
    t_think = _time.perf_counter()
    if steps > 0:
        avp_context = connector.think(
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
    text = connector.generate(
        prompt,
        context=avp_context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        _diagnostics=diagnostics,
    )
    generate_duration = _time.perf_counter() - t_gen

    logger.debug(
        "generate() model=%s steps=%d stored=%s prior=%s",
        model, steps, store_key, prior_key,
    )
    logger.info(
        "generate() think=%.3fs generate=%.3fs total=%.3fs",
        think_duration, generate_duration, _time.perf_counter() - t_start,
    )

    # Compare mode
    if debug_config is not None and debug_config.compare:
        _run_compare(diagnostics, connector, prompt, text, max_new_tokens, temperature)

    if diagnostics is not None:
        logger.info("generate() debug: %s", diagnostics.summary())

    if collect_metrics:
        from .metrics import GenerateMetrics

        metrics = GenerateMetrics(
            model=model,
            steps=steps,
            has_prior_context=context is not None,
            stored=stored,
            duration_s=_time.perf_counter() - t_start,
            think_duration_s=think_duration,
            generate_duration_s=generate_duration,
            diagnostics=diagnostics,
        )
        return text, metrics

    return text


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


def inspect(data: bytes) -> Dict[str, Any]:
    """Inspect an AVP binary payload without loading models.

    Decodes the header and metadata from raw AVP wire bytes. Useful for
    debugging wire payloads, logging, and observability tools.

    Args:
        data: Raw AVP binary bytes (from ``AVPContext.to_bytes()`` or wire).

    Returns:
        Dict with decoded header/metadata fields::

            {
                "version": 1,
                "flags": 3,
                "compressed": True,
                "has_map": False,
                "kv_cache": False,
                "payload_length": 12345,
                "metadata_length": 78,
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "hidden_dim": 3584,
                "num_layers": 28,
                "payload_type": "KV_CACHE",
                "dtype": "FLOAT32",
                "tensor_shape": [28, 2, 1, 128, 128],
                "mode": "LATENT",
                "session_id": "",
                "source_agent_id": "",
                "target_agent_id": "",
                "avp_map_id": "",
                "extra": {"model_hash": "abc123", ...},
                "raw_size": 12435,
            }

    Raises:
        DecodeError: If data is not valid AVP binary.
    """
    from .codec import decode as avp_decode

    msg = avp_decode(data)
    h = msg.header
    m = msg.metadata

    return {
        "version": h.version,
        "flags": h.flags,
        "compressed": bool(h.flags & 0x01),
        "has_map": bool(h.flags & 0x02),
        "kv_cache": bool(h.flags & 0x04),
        "payload_length": h.payload_length,
        "metadata_length": h.metadata_length,
        "model_id": m.model_id,
        "hidden_dim": m.hidden_dim,
        "num_layers": m.num_layers,
        "payload_type": m.payload_type.name if hasattr(m.payload_type, "name") else str(m.payload_type),
        "dtype": m.dtype.name if hasattr(m.dtype, "name") else str(m.dtype),
        "tensor_shape": list(m.tensor_shape) if m.tensor_shape else [],
        "mode": m.mode.name if hasattr(m.mode, "name") else str(m.mode),
        "session_id": m.session_id,
        "source_agent_id": m.source_agent_id,
        "target_agent_id": m.target_agent_id,
        "avp_map_id": m.avp_map_id,
        "extra": dict(m.extra) if m.extra else {},
        "raw_size": msg.raw_size,
    }
