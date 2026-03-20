"""Centralized vLLM imports and stub classes.

All vLLM import paths are gathered here so breakage from vLLM API changes
is caught in one place rather than scattered across multiple modules.

FRAGILE(vllm): F7, F8 — vLLM import paths change across versions.
"""

import logging

logger = logging.getLogger(__name__)

# --- vLLM KV connector base classes ---

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: F401
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )

    HAS_VLLM = True

    # Runtime version check — the model plugin uses FlashAttentionMetadata
    # and set_forward_context APIs validated on vLLM 0.17.x only.
    try:
        import vllm as _vllm_mod
        _vllm_ver = getattr(_vllm_mod, "__version__", "0.0.0")
        _major_minor = tuple(int(x) for x in _vllm_ver.split(".")[:2])
        if _major_minor < (0, 17):
            logger.warning(
                "vLLM %s detected; AVP model plugin requires >= 0.17.0. "
                "FlashAttentionMetadata API may be incompatible.",
                _vllm_ver,
            )
    except Exception:
        pass

except ImportError:
    HAS_VLLM = False

    class KVConnectorRole:
        """Stub for development without vLLM."""

        SCHEDULER = 0
        WORKER = 1

    class KVConnectorBase_V1:
        """Stub base class for development on non-Linux platforms."""

        def __init__(self, **kwargs):
            pass

    class KVConnectorMetadata:
        """Stub metadata class."""

        pass


# --- vLLM model classes ---

try:
    from vllm.model_executor.models.registry import ModelRegistry  # noqa: F401

    HAS_VLLM_MODELS = True
except ImportError:
    HAS_VLLM_MODELS = False

    class ModelRegistry:
        """Stub for development without vLLM."""

        @staticmethod
        def register_model(name, cls):
            pass


try:
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM  # noqa: F401

    HAS_QWEN2 = True
except ImportError:
    HAS_QWEN2 = False
    Qwen2ForCausalLM = None  # type: ignore[assignment,misc]
