"""AVP connector for Ollama-managed GGUF models.

Resolves Ollama model names (e.g., ``"qwen2.5:7b"``) to GGUF blob
paths on disk, auto-unloads the model from the Ollama server to free
VRAM, then delegates all inference to :class:`LlamaCppConnector`.

Usage::

    from avp.connectors.ollama import OllamaConnector

    connector = OllamaConnector.from_ollama("qwen2.5:7b")
    context = connector.think("Analyze this problem", steps=10)
    answer = connector.generate("Solve step by step", context=context)

Cross-model::

    researcher = OllamaConnector.from_ollama("qwen2.5:7b")
    solver = OllamaConnector.from_ollama("llama3.2:3b")
    context = researcher.think("Analyze this", steps=10)
    answer = solver.generate("Solve it", context=context,
                             source=researcher, cross_model=True)

Requires: ``pip install avp[ollama]`` (same deps as ``avp[llamacpp]``)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .llamacpp import LlamaCppConnector

logger = logging.getLogger(__name__)

# Default Ollama registry and namespace for short model names
_DEFAULT_REGISTRY = "registry.ollama.ai"
_DEFAULT_NAMESPACE = "library"

# GGUF model layer media type in Ollama manifests
_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


# ---------------------------------------------------------------------------
# Ollama model name → GGUF path resolution
# ---------------------------------------------------------------------------


def _get_models_dir() -> Path:
    """Return the Ollama models directory."""
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def resolve_ollama_model(model_name: str) -> str:
    """Resolve an Ollama model name to a GGUF blob path on disk.

    Supports short names (``"qwen2.5:7b"``), bare names without a tag
    (``"qwen2.5"`` → tag ``"latest"``), and fully-qualified names
    (``"registry.ollama.ai/library/qwen2.5:7b"``).

    Args:
        model_name: Ollama model name.

    Returns:
        Absolute path to the GGUF blob file.

    Raises:
        FileNotFoundError: If the model is not downloaded or the blob
            is missing.
        ValueError: If the manifest doesn't contain a model layer.
    """
    models_dir = _get_models_dir()
    name, tag = _parse_model_name(model_name)

    manifest_path = (
        models_dir / "manifests" / _DEFAULT_REGISTRY / _DEFAULT_NAMESPACE
        / name / tag
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ollama model '{model_name}' not found at {manifest_path}. "
            f"Run: ollama pull {model_name}"
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Find the GGUF model layer
    digest = None
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == _MODEL_MEDIA_TYPE:
            digest = layer.get("digest", "")
            break

    if not digest:
        raise ValueError(
            f"No model layer (mediaType={_MODEL_MEDIA_TYPE}) found in "
            f"manifest for '{model_name}'. Available layers: "
            f"{[l.get('mediaType') for l in manifest.get('layers', [])]}"
        )

    # Convert digest sha256:abc... → blob path sha256-abc...
    blob_name = digest.replace(":", "-")
    blob_path = models_dir / "blobs" / blob_name

    if not blob_path.exists():
        raise FileNotFoundError(
            f"GGUF blob not found at {blob_path}. "
            f"The Ollama model store may be corrupted. "
            f"Try: ollama pull {model_name}"
        )

    return str(blob_path.resolve())


def _parse_model_name(model_name: str) -> tuple:
    """Parse an Ollama model name into (name, tag).

    Examples:
        "qwen2.5"                                      → ("qwen2.5", "latest")
        "qwen2.5:7b"                                   → ("qwen2.5", "7b")
        "registry.ollama.ai/library/qwen2.5:7b"        → ("qwen2.5", "7b")
    """
    # Strip known registry/namespace prefix
    prefixes = [
        f"{_DEFAULT_REGISTRY}/{_DEFAULT_NAMESPACE}/",
        f"{_DEFAULT_NAMESPACE}/",
    ]
    for prefix in prefixes:
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix):]
            break

    # Split name:tag
    if ":" in model_name:
        name, tag = model_name.rsplit(":", 1)
    else:
        name, tag = model_name, "latest"

    return name, tag


# ---------------------------------------------------------------------------
# Ollama server interaction (best-effort, all failures are non-fatal)
# ---------------------------------------------------------------------------


def _get_ollama_host() -> str:
    """Return the Ollama API base URL."""
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def is_ollama_running() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        url = f"{_get_ollama_host()}/api/version"
        req = Request(url, method="GET")
        with urlopen(req, timeout=2):
            return True
    except (URLError, OSError, ValueError):
        return False


def is_model_loaded(model_name: str) -> bool:
    """Check if a model is currently loaded in the Ollama server."""
    try:
        url = f"{_get_ollama_host()}/api/ps"
        req = Request(url, method="GET")
        with urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())

        name, tag = _parse_model_name(model_name)
        name_lower = name.lower()
        tag_lower = tag.lower()
        target_variants = {
            f"{name_lower}:{tag_lower}",
            f"{_DEFAULT_NAMESPACE}/{name_lower}:{tag_lower}",
        }
        if tag_lower == "latest":
            target_variants.add(name_lower)
            target_variants.add(f"{_DEFAULT_NAMESPACE}/{name_lower}")

        for entry in data.get("models", []):
            entry_name = entry.get("name", "").lower()
            if entry_name in target_variants or entry_name.startswith(f"{name_lower}:"):
                return True

        return False
    except (URLError, OSError, ValueError, KeyError):
        return False


def unload_model(model_name: str) -> bool:
    """Unload a model from the Ollama server to free VRAM.

    Uses the documented ``keep_alive=0`` mechanism.
    """
    try:
        url = f"{_get_ollama_host()}/api/generate"
        body = json.dumps({
            "model": model_name,
            "prompt": "",
            "keep_alive": 0,
        }).encode()
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=10):
            pass
        logger.info("Unloaded '%s' from Ollama server", model_name)
        return True
    except (URLError, OSError, ValueError) as e:
        logger.warning("Failed to unload '%s' from Ollama: %s", model_name, e)
        return False


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class OllamaConnector(LlamaCppConnector):
    """AVP connector for Ollama-managed GGUF models.

    Resolves Ollama model names (e.g., ``"qwen2.5:7b"``) to GGUF blob
    paths on disk, auto-unloads the model from the Ollama server to
    free VRAM, then delegates all inference to
    :class:`LlamaCppConnector`.

    The auto-unload is non-destructive: Ollama will automatically
    reload the model on the next request to its API.
    """

    def __init__(
        self,
        model_name: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        auto_unload: bool = True,
        **kwargs: Any,
    ):
        self._ollama_model_name = model_name
        self._auto_unload = auto_unload

        # Step 1: Resolve Ollama name → GGUF blob path
        gguf_path = resolve_ollama_model(model_name)
        logger.info(
            "Resolved Ollama model '%s' → %s", model_name, gguf_path,
        )

        # Step 2: Auto-unload from Ollama server if running
        if auto_unload and is_ollama_running():
            if is_model_loaded(model_name):
                logger.info(
                    "Unloading '%s' from Ollama server to free VRAM",
                    model_name,
                )
                unload_model(model_name)

        # Step 3: Delegate to LlamaCppConnector
        super().__init__(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def from_ollama(
        cls,
        model_name: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        auto_unload: bool = True,
        **kwargs: Any,
    ) -> "OllamaConnector":
        """Load an Ollama model for latent communication.

        Resolves the model name to a GGUF blob, optionally unloads it
        from the Ollama server, and loads it via llama-cpp-python.

        Args:
            model_name: Ollama model name (e.g., ``"qwen2.5:7b"``).
            n_ctx: Context window size (default: 4096).
            n_gpu_layers: Number of layers to offload to GPU (-1 = all).
            verbose: Enable llama.cpp logging.
            auto_unload: Unload model from Ollama server before loading
                to avoid double memory usage (default: True).

        Returns:
            An OllamaConnector instance.
        """
        return cls(
            model_name=model_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            auto_unload=auto_unload,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        auto_unload: bool = True,
        **kwargs: Any,
    ) -> "OllamaConnector":
        """Alias for :meth:`from_ollama` (ABC compatibility)."""
        return cls.from_ollama(
            model_name=model_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            auto_unload=auto_unload,
            **kwargs,
        )

    def get_model_identity(self) -> Any:
        """Return model identity using the Ollama name, not blob path."""
        from ..types import ModelIdentity
        return ModelIdentity(
            model_id=self._ollama_model_name,
            hidden_size=self._n_embd,
            num_layers=self._n_layer or 0,
            vocab_size=self._n_vocab,
        )

    @staticmethod
    def list_models() -> list:
        """List Ollama models available locally.

        Returns:
            List of dicts with ``name`` and ``gguf_path`` keys, or
            an empty list if the models directory doesn't exist.
        """
        models_dir = _get_models_dir()
        manifests_dir = (
            models_dir / "manifests" / _DEFAULT_REGISTRY / _DEFAULT_NAMESPACE
        )
        if not manifests_dir.exists():
            return []

        results = []
        for model_dir in sorted(manifests_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for tag_file in sorted(model_dir.iterdir()):
                if tag_file.is_dir():
                    continue
                name = f"{model_dir.name}:{tag_file.name}"
                try:
                    path = resolve_ollama_model(name)
                    results.append({"name": name, "gguf_path": path})
                except (FileNotFoundError, ValueError):
                    pass
        return results
