"""AVP engine connectors for model inference backends."""

from .base import EngineConnector

__all__ = ["EngineConnector"]

# Lazy import for HuggingFaceConnector to avoid torch dependency
def __getattr__(name: str):
    if name == "HuggingFaceConnector":
        from .huggingface import HuggingFaceConnector
        return HuggingFaceConnector
    if name == "VLLMConnector":
        from .vllm import VLLMConnector
        return VLLMConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
