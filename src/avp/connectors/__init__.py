"""AVP engine connectors for model inference backends."""

from .base import EngineConnector

__all__ = [
    "EngineConnector",
    "HuggingFaceConnector",
    "VLLMConnector",
    "LlamaCppConnector",
    "OllamaConnector",
]

# Lazy imports to avoid heavy dependencies at module level
def __getattr__(name: str):
    if name == "HuggingFaceConnector":
        from .huggingface import HuggingFaceConnector
        return HuggingFaceConnector
    if name == "VLLMConnector":
        from .vllm import VLLMConnector
        return VLLMConnector
    if name == "LlamaCppConnector":
        from .llamacpp import LlamaCppConnector
        return LlamaCppConnector
    if name == "OllamaConnector":
        from .ollama import OllamaConnector
        return OllamaConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
