"""AVP KV-cache connector plugin for vLLM.

Implements KVConnectorBase_V1 to intercept KV-cache save/load in vLLM's
attention pipeline. Uses a file-based store (safetensors) for KV-cache
exchange between agents.

Loaded by vLLM at runtime via:
    LLM(
        model="...",
        kv_connector="avp.connectors.vllm_kv_connector.AVPKVConnectorV1Dynamic",
        kv_role="kv_both",
        kv_connector_extra_config={"avp_latent_steps": 10},
    )

FRAGILE(vllm): F1 — block_ids[0] single cache group
FRAGILE(vllm): F3 — kv_layer.shape[0]==2 K/V split
FRAGILE(vllm): F4 — layer.kv_cache[virtual_engine] access
FRAGILE(vllm): F5 — DBO mode attn_metadata as list
FRAGILE(vllm): F6 — scheduler_output.scheduled_new_reqs
"""

import hashlib
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

from ._vllm_compat import (
    HAS_VLLM,
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KVStore protocol + FileKVStore implementation
# ---------------------------------------------------------------------------

@runtime_checkable
class KVStore(Protocol):
    """Minimal protocol for KV-cache storage backends."""

    def save_layer(self, key: str, layer_idx: int, tensor: Any) -> None:
        """Save a single layer's KV tensor."""
        ...

    def load_layer(self, key: str, layer_idx: int) -> Optional[Any]:
        """Load a single layer's KV tensor. Returns None if not found."""
        ...

    def has_key(self, key: str) -> bool:
        """Check if any data exists for this key."""
        ...

    def get_seq_len(self, key: str) -> int:
        """Get the sequence length stored for this key. Returns 0 if not found."""
        ...

    def delete(self, key: str) -> None:
        """Delete all data for this key."""
        ...


class FileKVStore:
    """File-based KV store using safetensors for zero-copy tensor I/O.

    Directory layout:
        {store_dir}/{key}/layer_{idx}.safetensors
        {store_dir}/{key}/meta.txt  (seq_len)
    """

    def __init__(self, store_dir: str):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key_dir(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._dir / safe_key

    def save_layer(self, key: str, layer_idx: int, tensor: Any) -> None:
        import torch

        key_dir = self._key_dir(key)
        key_dir.mkdir(parents=True, exist_ok=True)
        path = key_dir / f"layer_{layer_idx}.pt"
        torch.save(tensor.cpu(), path)

    def load_layer(self, key: str, layer_idx: int) -> Optional[Any]:
        import torch

        path = self._key_dir(key) / f"layer_{layer_idx}.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def save_meta(self, key: str, seq_len: int, num_layers: int) -> None:
        """Save metadata (seq_len, num_layers) for a key."""
        key_dir = self._key_dir(key)
        key_dir.mkdir(parents=True, exist_ok=True)
        meta_path = key_dir / "meta.txt"
        meta_path.write_text(f"{seq_len}\n{num_layers}\n")

    def has_key(self, key: str) -> bool:
        meta = self._key_dir(key) / "meta.txt"
        return meta.exists()

    def get_seq_len(self, key: str) -> int:
        meta = self._key_dir(key) / "meta.txt"
        if not meta.exists():
            return 0
        try:
            return int(meta.read_text().strip().split("\n")[0])
        except (ValueError, IndexError):
            return 0

    def get_num_layers(self, key: str) -> int:
        """Get the number of layers stored for this key."""
        meta = self._key_dir(key) / "meta.txt"
        if not meta.exists():
            return 0
        try:
            lines = meta.read_text().strip().split("\n")
            return int(lines[1]) if len(lines) > 1 else 0
        except (ValueError, IndexError):
            return 0

    def delete(self, key: str) -> None:
        import shutil

        key_dir = self._key_dir(key)
        if key_dir.exists():
            shutil.rmtree(key_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AVPReqMeta:
    """Per-request metadata for KV connector operations."""

    request_id: str
    store_key: str
    num_tokens: int = 0
    num_external_tokens: int = 0
    block_ids: List[int] = field(default_factory=list)

    @classmethod
    def from_request(cls, request: Any, store_key: str = "") -> "AVPReqMeta":
        """Create from a vLLM request object."""
        req_id = str(getattr(request, "request_id", "default"))
        if not store_key:
            # Derive store key from prompt token IDs if available
            prompt_ids = getattr(request, "prompt_token_ids", None)
            if prompt_ids:
                store_key = compute_request_hash(prompt_ids)
            else:
                store_key = req_id
        return cls(request_id=req_id, store_key=store_key)


class AVPConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker each step."""

    def __init__(self, requests: Optional[List[AVPReqMeta]] = None):
        self.requests: List[AVPReqMeta] = requests or []


# ---------------------------------------------------------------------------
# KV Layout detection helpers
# ---------------------------------------------------------------------------

def _detect_kv_layout(tensor: Any) -> str:
    """Detect KV tensor layout from shape.

    FRAGILE(vllm): F2 — eliminated isinstance checks; shape-based only.

    Returns:
        "stacked_5d": [batch_or_blocks, 2, num_kv_heads, tokens, head_dim]
        "stacked_4d": [2, num_kv_heads, seq_len, head_dim]
        "unknown": unrecognized shape
    """
    ndim = tensor.dim() if hasattr(tensor, "dim") else len(tensor.shape)
    if ndim == 5:
        # FRAGILE(vllm): F3 — assert shape[1]==2 (K/V stacked)
        if tensor.shape[1] == 2:
            return "stacked_5d"
    elif ndim == 4:
        if tensor.shape[0] == 2:
            return "stacked_4d"
    return "unknown"


def _extract_kv_from_layer(tensor: Any) -> Tuple[Any, Any]:
    """Extract K and V tensors from a stacked KV layer tensor.

    Args:
        tensor: KV tensor in one of the supported layouts.

    Returns:
        Tuple of (K, V) tensors, each [num_kv_heads, seq_len, head_dim].

    Raises:
        ValueError: If the layout is not recognized.
    """
    layout = _detect_kv_layout(tensor)

    if layout == "stacked_5d":
        # [batch_or_blocks, 2, num_kv_heads, tokens, head_dim]
        if tensor.shape[0] == 1:
            # Single sequence: simple extraction
            k = tensor[0, 0]  # [num_kv_heads, tokens, head_dim]
            v = tensor[0, 1]
        else:
            # Multiple blocks: concatenate along token dimension.
            # NOTE: assumes blocks are in logical (not physical) order.
            # In vLLM's paged attention, the caller must reorder blocks
            # via block_table before passing to this function.
            # tensor[:, 0] -> [blocks, kv_heads, tokens, head_dim]
            # permute(1,0,2,3) -> [kv_heads, blocks, tokens, head_dim]
            # reshape -> [kv_heads, blocks*tokens, head_dim]
            k = tensor[:, 0].permute(1, 0, 2, 3).reshape(
                tensor.shape[2], -1, tensor.shape[4]
            )
            v = tensor[:, 1].permute(1, 0, 2, 3).reshape(
                tensor.shape[2], -1, tensor.shape[4]
            )
        return k, v

    elif layout == "stacked_4d":
        # [2, num_kv_heads, seq_len, head_dim]
        return tensor[0], tensor[1]

    else:
        raise ValueError(
            f"Unrecognized KV tensor layout: shape={tensor.shape}. "
            f"Expected 5D [B,2,H,T,D] or 4D [2,H,T,D]."
        )


def _inject_kv_into_layer(
    kv_cache: Any,
    layer_idx: int,
    k_tensor: Any,
    v_tensor: Any,
    slot_mapping: Optional[Any] = None,
) -> None:
    """Inject K and V tensors into a vLLM KV cache layer.

    Args:
        kv_cache: The vLLM KV cache tensor for this layer.
        layer_idx: Layer index (for logging).
        k_tensor: Key tensor [num_kv_heads, seq_len, head_dim].
        v_tensor: Value tensor [num_kv_heads, seq_len, head_dim].
        slot_mapping: Optional slot mapping for PagedAttention injection.
    """
    import torch

    if slot_mapping is not None:
        # PagedAttention: scatter into correct slots
        # kv_cache shape depends on backend, but typically:
        # [num_blocks, block_size, num_kv_heads, head_dim] for each of K and V
        # or [num_blocks, 2, num_kv_heads, block_size, head_dim]
        layout = _detect_kv_layout(kv_cache)

        seq_len = k_tensor.shape[1] if k_tensor.dim() == 3 else k_tensor.shape[0]
        device = kv_cache.device

        k_src = k_tensor.to(device)
        v_src = v_tensor.to(device)
        mapping = slot_mapping.to(device)

        # Ensure mapping covers our sequence
        effective_len = min(seq_len, mapping.shape[0])

        for i in range(effective_len):
            slot = mapping[i].item()
            if slot < 0:
                continue

            if layout == "stacked_5d":
                block_idx = slot // kv_cache.shape[3]
                offset = slot % kv_cache.shape[3]
                if k_src.dim() == 3:
                    kv_cache[block_idx, 0, :, offset, :] = k_src[:, i, :]
                    kv_cache[block_idx, 1, :, offset, :] = v_src[:, i, :]
            else:
                logger.warning(
                    "Unsupported KV cache layout for slot injection at layer %d",
                    layer_idx,
                )
                return
    else:
        # Direct copy (contiguous buffer)
        layout = _detect_kv_layout(kv_cache)
        if layout == "stacked_5d" and kv_cache.shape[0] == 1:
            device = kv_cache.device
            seq_len = k_tensor.shape[1] if k_tensor.dim() == 3 else k_tensor.shape[0]
            kv_cache[0, 0, :, :seq_len, :] = k_tensor.to(device)
            kv_cache[0, 1, :, :seq_len, :] = v_tensor.to(device)
        elif layout == "stacked_4d":
            device = kv_cache.device
            seq_len = k_tensor.shape[1] if k_tensor.dim() == 3 else k_tensor.shape[0]
            kv_cache[0, :, :seq_len, :] = k_tensor.to(device)
            kv_cache[1, :, :seq_len, :] = v_tensor.to(device)
        else:
            logger.warning(
                "Cannot inject into KV cache with layout %s at layer %d",
                layout, layer_idx,
            )


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------

class AVPKVConnectorV1Dynamic(KVConnectorBase_V1):
    """AVP KV-cache connector for vLLM.

    Intercepts save_kv_layer/start_load_kv calls in vLLM's attention pipeline
    to serialize/deserialize KV-cache for inter-agent transfer.

    KV-cache exchange uses a file-based store: each request's KV is stored
    per-layer in the configured store directory. The store key is derived from
    prompt token IDs so consumers can find pre-computed KV-cache from producers.

    Also bridges latent step configuration to the model plugin via the
    AVP_LATENT_STEPS environment variable.

    Configuration via kv_connector_extra_config:
        avp_latent_steps: Number of latent thinking steps (default: 10)
        avp_store_dir: Directory for KV-cache files (default: /tmp/avp_kv_store)
    """

    def __init__(self, vllm_config=None, role=None, kv_cache_config=None, **kwargs):
        # Extract extra_config before calling super
        self._extra_config: Dict[str, Any] = {}
        if vllm_config is not None:
            kv_config = getattr(vllm_config, "kv_transfer_config", None)
            if kv_config is not None:
                self._extra_config = getattr(kv_config, "kv_connector_extra_config", {}) or {}

        if HAS_VLLM and vllm_config is not None:
            # vLLM 0.17 passes (vllm_config, role, kv_cache_config) positionally
            init_kwargs = dict(vllm_config=vllm_config, role=role, **kwargs)
            if kv_cache_config is not None:
                init_kwargs["kv_cache_config"] = kv_cache_config
            super().__init__(**init_kwargs)
            self._role = role
        else:
            # Stub mode (testing without vLLM runtime)
            self._role = role

        # Bridge latent steps config to model plugin via env var
        latent_steps = self._extra_config.get("avp_latent_steps", 20)
        os.environ["AVP_LATENT_STEPS"] = str(latent_steps)

        # Initialize store
        store_dir = self._extra_config.get(
            "avp_store_dir",
            os.environ.get("AVP_KV_STORE_DIR", "/tmp/avp_kv_store"),
        )
        self._store = FileKVStore(store_dir)

        # Per-request state tracking
        self._pending_saves: Dict[str, Dict[int, Any]] = {}  # req_id → {layer_idx: tensor}
        self._pending_meta: Dict[str, int] = {}  # req_id → seq_len
        self._loaded_keys: Dict[str, str] = {}  # req_id → store_key
        self._lock = threading.RLock()

        # Reference to vLLM's KV cache buffers
        self._kv_caches: Optional[Dict[str, Any]] = None

        logger.info(
            "AVPKVConnectorV1Dynamic initialized: store=%s, latent_steps=%s",
            store_dir, latent_steps,
        )

    @property
    def role(self):
        if self._role is not None:
            return self._role
        return KVConnectorRole.WORKER

    def requires_piecewise_for_cudagraph(self) -> bool:
        """Tell vLLM that CUDA graphs need piecewise execution.

        This avoids CUDA graph capture issues with our KV injection.
        """
        return True

    # ----- Producer methods -----

    def save_kv_layer(
        self,
        layer_name: str,
        kv_tensor: Any,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        """Buffer a single layer's KV tensor during forward pass.

        Args:
            layer_name: Layer identifier (e.g., "model.layers.0.self_attn").
            kv_tensor: The KV tensor for this layer.
            attn_metadata: vLLM attention metadata (contains request info).
        """
        layer_idx = _parse_layer_index(layer_name)

        if layer_idx is None:
            logger.warning("Cannot parse layer index from %r -- skipping save", layer_name)
            return

        # Detect tensor layout
        layout = _detect_kv_layout(kv_tensor)

        if layout == "unknown":
            # vLLM 0.17 passes the entire paged KV buffer for the layer,
            # not a per-request slice. We cannot extract per-request data
            # without the block table. Log and skip.
            logger.debug(
                "Skipping save for %s: unrecognized layout shape=%s "
                "(likely full paged KV buffer)",
                layer_name, kv_tensor.shape,
            )
            return

        request_id = self._derive_store_key(attn_metadata)

        try:
            with self._lock:
                if request_id not in self._pending_saves:
                    self._pending_saves[request_id] = {}
                self._pending_saves[request_id][layer_idx] = kv_tensor.clone()

                # Track sequence length from tensor shape
                if layout == "stacked_5d":
                    seq_len = kv_tensor.shape[3]
                    if kv_tensor.shape[0] > 1:
                        seq_len = kv_tensor.shape[0] * kv_tensor.shape[3]
                elif layout == "stacked_4d":
                    seq_len = kv_tensor.shape[2]
                else:
                    seq_len = 0
                self._pending_meta[request_id] = seq_len
        except Exception as e:
            logger.warning("Failed to save KV layer %s for %s: %s", layer_name, request_id, e)

    def wait_for_save(self) -> None:
        """Flush all buffered layers to the store.

        Called after all layers have been saved for a request.
        """
        with self._lock:
            request_ids = list(self._pending_saves.keys())

        for request_id in request_ids:
            self._flush_to_store(request_id)

    def _flush_to_store(self, request_id: str) -> None:
        """Serialize buffered layers for a request to the file store."""
        with self._lock:
            layers = self._pending_saves.pop(request_id, None)
            seq_len = self._pending_meta.pop(request_id, 0)

        if not layers:
            return

        store_key = request_id
        num_layers = len(layers)

        try:
            for layer_idx, tensor in sorted(layers.items()):
                k, v = _extract_kv_from_layer(tensor)
                import torch

                # Stack as [2, num_kv_heads, seq_len, head_dim]
                stacked = torch.stack([k, v], dim=0)
                self._store.save_layer(store_key, layer_idx, stacked)

            self._store.save_meta(store_key, seq_len, num_layers)

            logger.debug(
                "Flushed KV for request %s: %d layers, seq_len=%d",
                request_id, num_layers, seq_len,
            )
        except Exception as e:
            logger.warning("Failed to flush KV for request %s: %s", request_id, e)

    # ----- Consumer methods -----

    def start_load_kv(self, forward_context: Any = None, **kwargs) -> None:
        """Check store for matching KV-cache and prepare for loading.

        Called before the forward pass to pre-load KV-cache from a producer.

        Args:
            forward_context: vLLM forward context with request information.
        """
        if forward_context is None:
            return

        store_key = self._derive_store_key(forward_context)
        if not store_key:
            return

        if self._store.has_key(store_key):
            request_id = self._extract_request_id(forward_context)
            with self._lock:
                self._loaded_keys[request_id] = store_key
            logger.debug("Found KV data for request %s (key=%s)", request_id, store_key)

    def wait_for_layer_load(
        self,
        layer_name: str,
        **kwargs,
    ) -> Optional[Any]:
        """Return a specific layer's KV data from the store.

        Args:
            layer_name: Layer identifier.

        Returns:
            KV tensor [2, num_kv_heads, seq_len, head_dim], or None.
        """
        layer_idx = _parse_layer_index(layer_name)
        if layer_idx is None:
            return None

        with self._lock:
            loaded = dict(self._loaded_keys)

        for request_id, store_key in loaded.items():
            tensor = self._store.load_layer(store_key, layer_idx)
            if tensor is not None:
                return tensor

        return None

    def request_finished(
        self,
        request: Any,
        block_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Called when a request finishes.

        Returns:
            Tuple of (should_free_blocks, metadata).
            should_free_blocks=False: we may want to keep KV for consumers.
        """
        store_key = self._derive_store_key(request)

        # Flush any pending save data
        with self._lock:
            if store_key in self._pending_saves:
                self._flush_to_store(store_key)

        # Clean up loaded state
        with self._lock:
            self._loaded_keys.pop(store_key, None)

        return (False, None)

    # ----- Scheduler methods (Phase 2) -----

    def build_connector_meta(self, scheduler_output: Any) -> "KVConnectorMetadata":
        """Build connector metadata for this scheduler step.

        Inspects scheduled requests and creates AVPReqMeta for each that
        has matching KV data in the store.

        FRAGILE(vllm): F6 — scheduler_output.scheduled_new_reqs
        """
        req_metas = []

        try:
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
            if new_reqs:
                for req in new_reqs:
                    meta = AVPReqMeta.from_request(req)
                    if self._store.has_key(meta.store_key):
                        meta.num_external_tokens = self._store.get_seq_len(meta.store_key)
                    req_metas.append(meta)
        except Exception as e:
            logger.warning("Failed to build connector meta: %s", e)

        return AVPConnectorMetadata(requests=req_metas)

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
        **kwargs,
    ) -> Tuple[int, bool]:
        """Check how many tokens from the store match this request.

        Returns:
            Tuple of (num_matched_tokens, is_async). is_async=False since
            our file-based loading is synchronous.
        """
        meta = AVPReqMeta.from_request(request)
        seq_len = self._store.get_seq_len(meta.store_key)

        if seq_len <= 0:
            return (0, False)

        matched = max(0, seq_len - num_computed_tokens)
        return (matched, False)

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int,
    ) -> None:
        """Update state after block allocation.

        Records the block IDs allocated for this request so we can inject
        KV data into the correct slots.
        """
        if num_external_tokens <= 0:
            return

        meta = AVPReqMeta.from_request(request)

        # FRAGILE(vllm): F1 — assume single cache group
        try:
            if hasattr(blocks, "__getitem__"):
                block_list = blocks[0] if len(blocks) > 0 else []
            else:
                block_list = list(blocks) if blocks else []
        except (TypeError, IndexError):
            block_list = []

        meta.block_ids = list(block_list)
        meta.num_external_tokens = num_external_tokens

    # ----- Registration and stats -----

    def register_kv_caches(self, kv_caches: Dict[str, Any]) -> None:
        """Store reference to vLLM's GPU KV cache buffers."""
        self._kv_caches = kv_caches
        logger.debug("Registered %d KV cache layers", len(kv_caches))

    # ----- Stub methods (no-op for Phase 1) -----

    def handle_preemptions(self, **kwargs) -> None:
        """Handle request preemptions (no-op)."""
        pass

    def get_block_ids_with_load_errors(self, **kwargs) -> Set[int]:
        """Return block IDs that failed to load (none for Phase 1)."""
        return set()

    def get_kv_connector_stats(self, **kwargs) -> Any:
        """Return connector statistics.

        vLLM 0.17 expects KVConnectorStats | None. Returning None
        opts out of stats aggregation.
        """
        return None

    # ----- Helpers -----

    def _extract_request_id(self, obj: Any) -> str:
        """Extract a request identifier from various vLLM objects.

        Works with attention metadata, request objects, and forward contexts.
        """
        # Direct request_id attribute
        if hasattr(obj, "request_id"):
            return str(obj.request_id)

        # Forward context with requests list
        if hasattr(obj, "requests") and obj.requests:
            first = obj.requests[0]
            if hasattr(first, "request_id"):
                return str(first.request_id)

        # FRAGILE(vllm): F5 -- DBO mode attn_metadata as list
        if isinstance(obj, list) and obj:
            return self._extract_request_id(obj[0])

        # String passthrough
        if isinstance(obj, str):
            return obj

        return "default"

    def _derive_store_key(self, obj: Any) -> str:
        """Derive a consistent store key from a vLLM object.

        Uses prompt_token_ids hash when available (content-addressable),
        falls back to request_id. Both producer and consumer must use
        this method to ensure matching keys.
        """
        # Try prompt_token_ids for content-addressable lookup
        prompt_ids = getattr(obj, "prompt_token_ids", None)
        if prompt_ids:
            return compute_request_hash(prompt_ids)

        # Forward context with requests list
        if hasattr(obj, "requests") and obj.requests:
            first = obj.requests[0]
            prompt_ids = getattr(first, "prompt_token_ids", None)
            if prompt_ids:
                return compute_request_hash(prompt_ids)

        # Fall back to request_id
        return self._extract_request_id(obj)


# ---------------------------------------------------------------------------
# Module-level helpers (kept public for backward compatibility)
# ---------------------------------------------------------------------------

def _parse_layer_index(layer_name: str) -> Optional[int]:
    """Extract numeric layer index from a layer name.

    E.g., "model.layers.5.self_attn" -> 5
    """
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    return None


def compute_request_hash(token_ids: List[int]) -> str:
    """Compute a hash of prompt token IDs for store lookup.

    This allows consumers to find pre-computed KV-cache from producers
    that processed the same prompt.

    Args:
        token_ids: List of prompt token IDs.

    Returns:
        Hex-encoded SHA-256 hash string (first 16 chars).
    """
    data = ",".join(str(t) for t in token_ids).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]
