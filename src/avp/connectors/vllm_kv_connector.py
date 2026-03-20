"""AVP KV-cache connector plugin for vLLM.

Implements KVConnectorBase_V1 for multi-agent KV-cache transfer.
Agent A's request produces a KV-cache that is saved to a file-based store.
Agent B's request loads it as a prefix and generates from Agent A's computation.

The connector handles per-request KV extraction from vLLM's paged attention
buffer using block_ids and slot_mapping, following the same pattern as
vLLM's ExampleConnector.

FRAGILE(vllm): F1 -- block_ids[0] single cache group
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
    def save_layer(self, key: str, layer_idx: int, tensor: Any) -> None: ...
    def load_layer(self, key: str, layer_idx: int) -> Optional[Any]: ...
    def has_key(self, key: str) -> bool: ...
    def get_seq_len(self, key: str) -> int: ...
    def delete(self, key: str) -> None: ...


class FileKVStore:
    """File-based KV store using torch.save for tensor I/O."""

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
        torch.save(tensor.cpu(), key_dir / f"layer_{layer_idx}.pt")

    def load_layer(self, key: str, layer_idx: int) -> Optional[Any]:
        import torch
        path = self._key_dir(key) / f"layer_{layer_idx}.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def save_meta(self, key: str, seq_len: int, num_layers: int) -> None:
        key_dir = self._key_dir(key)
        key_dir.mkdir(parents=True, exist_ok=True)
        (key_dir / "meta.txt").write_text(f"{seq_len}\n{num_layers}\n")

    def has_key(self, key: str) -> bool:
        return (self._key_dir(key) / "meta.txt").exists()

    def get_seq_len(self, key: str) -> int:
        meta = self._key_dir(key) / "meta.txt"
        if not meta.exists():
            return 0
        try:
            return int(meta.read_text().strip().split("\n")[0])
        except (ValueError, IndexError):
            return 0

    def get_num_layers(self, key: str) -> int:
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
# Metadata
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
        req_id = str(getattr(request, "request_id", "default"))
        if not store_key:
            prompt_ids = getattr(request, "prompt_token_ids", None)
            if prompt_ids:
                store_key = compute_request_hash(prompt_ids)
            else:
                store_key = req_id
        return cls(request_id=req_id, store_key=store_key)


class AVPConnectorMetadata(KVConnectorMetadata):
    def __init__(self, requests: Optional[List[AVPReqMeta]] = None):
        self.requests: List[AVPReqMeta] = requests or []


# ---------------------------------------------------------------------------
# Slot mapping helpers
# ---------------------------------------------------------------------------

def _compute_slot_mapping(block_ids: List[int], block_size: int, num_tokens: int) -> Any:
    """Compute physical slot indices from block IDs.

    slot = block_id * block_size + offset_within_block

    Following the ExampleConnector pattern from vLLM.
    """
    import torch

    block_ids_t = torch.tensor(block_ids, dtype=torch.long)
    offsets = torch.arange(block_size, dtype=torch.long)
    mapping = block_ids_t.reshape(-1, 1) * block_size + offsets.reshape(1, -1)
    return mapping.flatten()[:num_tokens]


def _extract_request_kv(kv_cache: Any, slot_mapping: Any) -> Any:
    """Extract per-request KV entries from the full paged buffer.

    Args:
        kv_cache: Full paged buffer [2, num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: Physical slot indices [num_tokens]

    Returns:
        Per-request KV [2, num_tokens, num_kv_heads * head_dim]
    """
    slot_mapping = slot_mapping.to(kv_cache.device)
    num_pages = kv_cache.shape[1]
    page_size = kv_cache.shape[2]
    flat = kv_cache.reshape(2, num_pages * page_size, -1)
    return flat[:, slot_mapping, :].clone()


def _inject_request_kv(kv_cache: Any, slot_mapping: Any, kv_data: Any) -> None:
    """Inject per-request KV entries into the paged buffer.

    Args:
        kv_cache: Full paged buffer [2, num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: Physical slot indices [num_tokens]
        kv_data: Per-request KV [2, num_tokens, num_kv_heads * head_dim]
    """
    slot_mapping = slot_mapping.to(kv_cache.device)
    kv_data = kv_data.to(kv_cache.device)
    num_pages = kv_cache.shape[1]
    page_size = kv_cache.shape[2]
    flat = kv_cache.reshape(2, num_pages * page_size, -1)
    flat[:, slot_mapping, :] = kv_data


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------

class AVPKVConnectorV1Dynamic(KVConnectorBase_V1):
    """AVP KV-cache connector for multi-agent transfer via vLLM.

    Agent A's request: model plugin runs latent steps, request finishes,
    connector extracts per-request KV from paged buffer and saves to store.

    Agent B's request: connector detects matching KV in store, tells scheduler
    about external tokens, injects stored KV into allocated blocks before
    forward pass.

    Configuration via kv_connector_extra_config:
        avp_latent_steps: Number of latent thinking steps (default: 20)
        avp_store_dir: Directory for KV-cache files (default: /tmp/avp_kv_store)
    """

    def __init__(self, vllm_config=None, role=None, kv_cache_config=None, **kwargs):
        self._extra_config: Dict[str, Any] = {}
        if vllm_config is not None:
            kv_config = getattr(vllm_config, "kv_transfer_config", None)
            if kv_config is not None:
                self._extra_config = getattr(kv_config, "kv_connector_extra_config", {}) or {}

        if HAS_VLLM and vllm_config is not None:
            init_kwargs = dict(vllm_config=vllm_config, role=role, **kwargs)
            if kv_cache_config is not None:
                init_kwargs["kv_cache_config"] = kv_cache_config
            super().__init__(**init_kwargs)
            self._role = role
        else:
            self._role = role

        # Bridge latent steps config to model plugin
        latent_steps = self._extra_config.get("avp_latent_steps", 20)
        os.environ["AVP_LATENT_STEPS"] = str(latent_steps)

        # Store
        store_dir = self._extra_config.get(
            "avp_store_dir",
            os.environ.get("AVP_KV_STORE_DIR", "/tmp/avp_kv_store"),
        )
        self._store = FileKVStore(store_dir)

        # Block size for slot mapping computation
        self._block_size = 16
        if vllm_config is not None:
            cache_config = getattr(vllm_config, "cache_config", None)
            if cache_config is not None:
                self._block_size = getattr(cache_config, "block_size", 16)

        # GPU KV cache buffers (set via register_kv_caches)
        self._kv_caches: Optional[Dict[str, Any]] = None

        # Pending save data (from worker save_kv_layer → wait_for_save)
        self._pending_saves: Dict[str, Dict] = {}
        # Pending load metadata (from scheduler → worker)
        self._pending_loads: Dict[str, AVPReqMeta] = {}
        self._lock = threading.RLock()

        logger.info(
            "AVPKVConnectorV1Dynamic initialized: store=%s, block_size=%d",
            store_dir, self._block_size,
        )

    @property
    def role(self):
        if self._role is not None:
            return self._role
        return KVConnectorRole.WORKER

    def requires_piecewise_for_cudagraph(self) -> bool:
        return True

    # ----- Save side (Agent A finishes → KV extracted and saved) -----

    def save_kv_layer(self, layer_name: str, kv_tensor: Any,
                      attn_metadata: Any, **kwargs) -> None:
        """Extract per-request KV from the paged buffer during forward.

        This runs on the WORKER side where register_kv_caches was called
        and GPU buffers are accessible. Uses attn_metadata to identify
        which request's tokens to extract via slot_mapping.
        """
        layer_idx = _parse_layer_index(layer_name)
        if layer_idx is None:
            return

        # Get block_table and seq_lens from attn_metadata to extract per-request KV
        try:
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
            fwd_meta = getattr(ctx, "attn_metadata", None)
            if fwd_meta is None:
                return

            # Get per-layer metadata
            if isinstance(fwd_meta, dict):
                layer_meta = fwd_meta.get(layer_name)
                if layer_meta is None:
                    return
            else:
                layer_meta = fwd_meta

            block_table = getattr(layer_meta, "block_table", None)
            seq_lens = getattr(layer_meta, "seq_lens", None)
            if block_table is None or seq_lens is None:
                return

            # For single-request batches, extract the first request's KV
            num_reqs = int((seq_lens > 0).sum().item()) if seq_lens is not None else 0
            if num_reqs != 1:
                return  # Only save for single-request batches

            num_tokens = int(seq_lens[0].item())
            if num_tokens <= 0:
                return

            # Compute block_ids from block_table[0]
            num_blocks_needed = (num_tokens + self._block_size - 1) // self._block_size
            block_ids = block_table[0, :num_blocks_needed].tolist()

            slot_mapping = _compute_slot_mapping(block_ids, self._block_size, num_tokens)
            per_request_kv = _extract_request_kv(kv_tensor, slot_mapping)

            # Derive store key from request_id in forward context
            request_id = self._extract_request_id(ctx)
            store_key = request_id

            with self._lock:
                if store_key not in self._pending_saves:
                    self._pending_saves[store_key] = {"num_tokens": num_tokens}
                self._pending_saves[store_key][layer_idx] = per_request_kv

        except Exception as e:
            if not getattr(self, "_save_error_logged", False):
                logger.warning("save_kv_layer error: %s", e)
                self._save_error_logged = True

    def wait_for_save(self) -> None:
        """Flush all pending saves to the store."""
        with self._lock:
            pending = dict(self._pending_saves)
            self._pending_saves.clear()

        for store_key, layers_data in pending.items():
            num_tokens = layers_data.pop("num_tokens", 0)
            num_layers = 0
            try:
                for layer_idx, kv_data in sorted(layers_data.items()):
                    self._store.save_layer(store_key, layer_idx, kv_data)
                    num_layers += 1
                if num_layers > 0:
                    self._store.save_meta(store_key, num_tokens, num_layers)
                    logger.info(
                        "Saved KV for %s: %d layers, %d tokens",
                        store_key, num_layers, num_tokens,
                    )
            except Exception as e:
                logger.warning("Failed to flush KV for %s: %s", store_key, e)

    def request_finished(
        self, request: Any, block_ids: Optional[List[int]] = None, **kwargs,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Called when a request finishes. Triggers flush of pending saves."""
        # Flush any pending layer data from save_kv_layer
        self.wait_for_save()
        return (True, None)

    # ----- Load side (Agent B starts → KV injected from store) -----

    def start_load_kv(self, forward_context: Any = None, **kwargs) -> None:
        """Inject stored KV into the paged buffer before forward pass.

        Uses pending load metadata (block_ids from scheduler) to compute
        slot_mapping, then writes stored KV into the allocated blocks.
        """
        if forward_context is None or self._kv_caches is None:
            return

        with self._lock:
            pending = dict(self._pending_loads)

        for req_id, meta in pending.items():
            if not self._store.has_key(meta.store_key):
                continue

            slot_mapping = _compute_slot_mapping(
                meta.block_ids, self._block_size, meta.num_external_tokens,
            )

            layers_loaded = 0
            for layer_name, kv_cache in self._kv_caches.items():
                layer_idx = _parse_layer_index(layer_name)
                if layer_idx is None:
                    continue

                stored_kv = self._store.load_layer(meta.store_key, layer_idx)
                if stored_kv is None:
                    continue

                _inject_request_kv(kv_cache, slot_mapping, stored_kv)
                layers_loaded += 1

            if layers_loaded > 0:
                logger.debug(
                    "Loaded KV for %s: %d layers, %d tokens",
                    meta.store_key, layers_loaded, meta.num_external_tokens,
                )

        with self._lock:
            self._pending_loads.clear()

    def wait_for_layer_load(self, layer_name: str, **kwargs) -> Optional[Any]:
        """No-op — injection happens in start_load_kv."""
        return None

    # ----- Scheduler methods -----

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int, **kwargs,
    ) -> Tuple[int, bool]:
        """Tell the scheduler how many tokens we can provide from the store.

        If Agent A's KV is in the store for this prompt, return the token
        count. The scheduler will allocate blocks and mark them as external.
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
        """Record block allocation for pending load.

        The scheduler allocated blocks for external tokens. We save the
        block_ids so start_load_kv can inject KV into the right slots.
        """
        if num_external_tokens <= 0:
            return

        meta = AVPReqMeta.from_request(request)

        # FRAGILE(vllm): F1 -- single cache group
        try:
            if hasattr(blocks, "get_block_ids"):
                block_list = list(blocks.get_block_ids()[0])
            elif hasattr(blocks, "__getitem__"):
                block_list = list(blocks[0]) if len(blocks) > 0 else []
            else:
                block_list = list(blocks) if blocks else []
        except (TypeError, IndexError):
            block_list = []

        meta.block_ids = block_list
        meta.num_external_tokens = num_external_tokens

        with self._lock:
            self._pending_loads[meta.request_id] = meta

        logger.debug(
            "Scheduled load for %s: %d external tokens, %d blocks",
            meta.store_key, num_external_tokens, len(block_list),
        )

    def build_connector_meta(self, scheduler_output: Any) -> "KVConnectorMetadata":
        return AVPConnectorMetadata()

    # ----- Registration and stats -----

    def register_kv_caches(self, kv_caches: Dict[str, Any]) -> None:
        """Store reference to vLLM's GPU KV cache buffers.

        These are used in request_finished (extraction) and
        start_load_kv (injection).
        """
        self._kv_caches = kv_caches
        logger.info(
            "Registered %d KV cache layers: %s",
            len(kv_caches),
            list(kv_caches.keys())[:3],
        )

    def handle_preemptions(self, **kwargs) -> None:
        pass

    def get_block_ids_with_load_errors(self, **kwargs) -> Set[int]:
        return set()

    def get_kv_connector_stats(self, **kwargs) -> Any:
        return None

    # ----- Helpers -----

    def _extract_request_id(self, obj: Any) -> str:
        if hasattr(obj, "request_id"):
            return str(obj.request_id)
        if hasattr(obj, "requests") and obj.requests:
            first = obj.requests[0]
            if hasattr(first, "request_id"):
                return str(first.request_id)
        if isinstance(obj, list) and obj:
            return self._extract_request_id(obj[0])
        if isinstance(obj, str):
            return obj
        return "default"

    def _derive_store_key(self, obj: Any) -> str:
        prompt_ids = getattr(obj, "prompt_token_ids", None)
        if prompt_ids:
            return compute_request_hash(prompt_ids)
        if hasattr(obj, "requests") and obj.requests:
            first = obj.requests[0]
            prompt_ids = getattr(first, "prompt_token_ids", None)
            if prompt_ids:
                return compute_request_hash(prompt_ids)
        return self._extract_request_id(obj)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_layer_index(layer_name: str) -> Optional[int]:
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    return None


def compute_request_hash(token_ids: List[int]) -> str:
    data = ",".join(str(t) for t in token_ids).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]
