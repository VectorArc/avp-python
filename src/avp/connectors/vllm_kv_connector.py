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

# Module-level shared state between scheduler and worker connector instances.
# vLLM creates separate SCHEDULER and WORKER connector instances. In
# UniProcExecutor they share a process; in MultiProcExecutor they don't.
# We use module-level dicts (works for UniProc) with FileKVStore fallback
# for cross-process scenarios.

# request_id → store_key (prompt hash). Scheduler sets, worker reads.
_REQUEST_STORE_KEYS: Dict[str, str] = {}

# request_id → AVPReqMeta for pending loads. Scheduler sets via
# update_state_after_alloc, worker reads via start_load_kv.
_PENDING_LOADS: Dict[str, Any] = {}

# store_key → projected embedding tensor (CPU). Model plugin sets after
# cross-model rosetta projection, connector flushes to FileKVStore.
_PROJECTED_EMBEDDINGS: Dict[str, Any] = {}


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
        target = key_dir / f"layer_{layer_idx}.pt"
        tmp = target.with_suffix(".pt.tmp")
        torch.save(tensor.cpu(), tmp)
        os.rename(str(tmp), str(target))

    def load_layer(self, key: str, layer_idx: int) -> Optional[Any]:
        import torch
        path = self._key_dir(key) / f"layer_{layer_idx}.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def save_meta(self, key: str, seq_len: int, num_layers: int) -> None:
        key_dir = self._key_dir(key)
        key_dir.mkdir(parents=True, exist_ok=True)
        target = key_dir / "meta.txt"
        tmp = target.with_suffix(".txt.tmp")
        tmp.write_text(f"{seq_len}\n{num_layers}\n")
        os.rename(str(tmp), str(target))

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

    def save_projected(self, key: str, tensor: Any) -> None:
        """Save a projected embedding for cross-model transfer."""
        import torch
        key_dir = self._key_dir(key)
        key_dir.mkdir(parents=True, exist_ok=True)
        target = key_dir / "projected.pt"
        tmp = target.with_suffix(".pt.tmp")
        torch.save(tensor.cpu(), tmp)
        os.rename(str(tmp), str(target))

    def load_projected(self, key: str) -> Optional[Any]:
        """Load a projected embedding for cross-model transfer."""
        import torch
        path = self._key_dir(key) / "projected.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def has_projected(self, key: str) -> bool:
        """Check if a projected embedding exists for this key."""
        return (self._key_dir(key) / "projected.pt").exists()

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

    Requires UniProcExecutor (single-process vLLM). Module-level shared
    state (``_REQUEST_STORE_KEYS``, ``_PENDING_LOADS``, ``_PROJECTED_EMBEDDINGS``)
    is not process-safe and will not work with MultiProcExecutor.

    Configuration via kv_connector_extra_config:
        avp_latent_steps: Number of latent thinking steps (default: 20)
        avp_store_dir: Directory for KV-cache files (default: /tmp/avp_kv_store)
        avp_target_model: Target model for cross-model rosetta (optional)
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

        # Bridge config to model plugin via env vars. vLLM creates exactly
        # one connector per role, so the process-wide env vars are safe.
        # Note: the model plugin's __init__ may run BEFORE the connector's
        # __init__ (vLLM loads the model first). Callers should pre-set
        # these env vars before creating the engine as a workaround.
        latent_steps = self._extra_config.get("avp_latent_steps", 20)
        os.environ["AVP_LATENT_STEPS"] = str(latent_steps)

        target_model = self._extra_config.get("avp_target_model", "")
        if target_model:
            os.environ["AVP_TARGET_MODEL"] = target_model
        else:
            os.environ.pop("AVP_TARGET_MODEL", None)

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
        self._save_error_count = 0
        self._save_thread: Optional[threading.Thread] = None

        # Cross-model mode: skip KV layer saves (useless to a different model).
        # The projected embedding is saved synchronously in wait_for_save.
        self._cross_model_only = bool(target_model)

        logger.info(
            "AVPKVConnectorV1Dynamic initialized: store=%s, block_size=%d, "
            "latent_steps=%s, cross_model=%s",
            store_dir, self._block_size, latent_steps,
            target_model or "(none)",
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

        Handles multi-request batches: iterates over all prefill requests
        (query_len > 1) and extracts each one's KV independently. Decode
        requests (query_len == 1) are skipped. Extracted data is queued
        in memory (_pending_saves) and flushed to disk asynchronously
        by wait_for_save.

        Skipped entirely in cross-model mode — the full KV cache from model A
        is useless to model B. Only the projected embedding matters.
        """
        if self._cross_model_only:
            return

        layer_idx = _parse_layer_index(layer_name)
        if layer_idx is None:
            return

        try:
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
            fwd_meta = getattr(ctx, "attn_metadata", None)
            if fwd_meta is None:
                return

            if isinstance(fwd_meta, dict):
                layer_meta = fwd_meta.get(layer_name)
                if layer_meta is None:
                    return
            else:
                layer_meta = fwd_meta

            # Only save during prefill batches
            max_query_len = getattr(layer_meta, "max_query_len", 1)
            if max_query_len <= 1:
                return

            block_table = getattr(layer_meta, "block_table", None)
            seq_lens = getattr(layer_meta, "seq_lens", None)
            query_start_loc = getattr(layer_meta, "query_start_loc", None)
            if block_table is None or seq_lens is None:
                return

            num_reqs = seq_lens.shape[0]

            # Iterate over all requests, extract prefill ones
            for req_idx in range(num_reqs):
                num_tokens = int(seq_lens[req_idx].item())
                if num_tokens <= 0:
                    continue

                # Skip decode requests (query_len == 1) in mixed batches
                if query_start_loc is not None and req_idx + 1 < query_start_loc.shape[0]:
                    query_len = int(query_start_loc[req_idx + 1].item()) - int(
                        query_start_loc[req_idx].item()
                    )
                    if query_len <= 1:
                        continue

                # Extract this request's KV
                num_blocks_needed = (num_tokens + self._block_size - 1) // self._block_size
                block_ids = block_table[req_idx, :num_blocks_needed].tolist()
                slot_mapping = _compute_slot_mapping(block_ids, self._block_size, num_tokens)
                # GPU→CPU copy (fast, ~1ms per layer for 7B)
                per_request_kv = _extract_request_kv(kv_tensor, slot_mapping)

                # Derive store key. For the first request, try the module-level
                # mapping. For subsequent requests in the batch, use a block-
                # table-derived key (unique per request).
                if req_idx == 0:
                    request_id = self._extract_request_id(ctx)
                    store_key = _REQUEST_STORE_KEYS.get(request_id, request_id)
                else:
                    # Hash block_ids as a unique key for this request
                    store_key = compute_request_hash(block_ids)

                with self._lock:
                    if store_key not in self._pending_saves:
                        self._pending_saves[store_key] = {"num_tokens": num_tokens}
                    self._pending_saves[store_key][layer_idx] = per_request_kv

        except Exception as e:
            self._save_error_count += 1
            lvl = logging.WARNING if self._save_error_count <= 3 else logging.DEBUG
            logger.log(lvl, "save_kv_layer error (#%d): %s", self._save_error_count, e)

    def wait_for_save(self) -> None:
        """Flush all pending saves to the store in a background thread."""
        # Join any still-running previous save thread to avoid two
        # threads writing to the same store key concurrently.
        prev = self._save_thread
        if prev is not None and prev.is_alive():
            prev.join(timeout=10.0)

        with self._lock:
            pending = dict(self._pending_saves)
            self._pending_saves.clear()

            # Also drain projected embeddings from the model plugin.
            # Under the same lock for consistency, though in UniProcExecutor
            # the model forward and wait_for_save are sequential.
            projected = dict(_PROJECTED_EMBEDDINGS)
            _PROJECTED_EMBEDDINGS.clear()
            if projected:
                logger.debug(
                    "wait_for_save: draining %d projected embeddings",
                    len(projected),
                )

        if not pending and not projected:
            return

        # Flush projected embeddings synchronously — they're tiny (~6KB,
        # <1ms) and callers need them immediately after generate() returns.
        for store_key, emb_tensor in projected.items():
            try:
                self._store.save_projected(store_key, emb_tensor)
                logger.debug(
                    "Saved projected embedding for %s: shape=%s",
                    store_key, list(emb_tensor.shape),
                )
            except Exception as e:
                logger.warning(
                    "Failed to save projected embedding for %s: %s",
                    store_key, e,
                )

        if not pending:
            return

        # KV layers are large (28 layers × MBs) — flush in background
        def _flush():
            for store_key, layers_data in pending.items():
                num_tokens = layers_data.pop("num_tokens", 0)
                num_layers = 0
                try:
                    for lid, kv_data in sorted(layers_data.items()):
                        if not isinstance(lid, int):
                            continue
                        self._store.save_layer(store_key, lid, kv_data)
                        num_layers += 1
                    if num_layers > 0:
                        self._store.save_meta(store_key, num_tokens, num_layers)
                        logger.debug(
                            "Saved KV for %s: %d layers, %d tokens",
                            store_key, num_layers, num_tokens,
                        )
                except Exception as e:
                    logger.warning("Failed to flush KV for %s: %s", store_key, e)

        # Run in background thread to avoid blocking the model runner
        t = threading.Thread(target=_flush, daemon=True)
        t.start()
        # Store thread reference so request_finished can wait if needed
        self._save_thread = t

    def request_finished(
        self, request: Any, block_ids: Optional[List[int]] = None, **kwargs,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Called when a request finishes. Ensures pending saves complete."""
        # Trigger flush if there's pending data
        self.wait_for_save()
        # Wait for background save thread if one is running
        save_thread = getattr(self, "_save_thread", None)
        if save_thread is not None and save_thread.is_alive():
            save_thread.join(timeout=5.0)
        # Clean up request_id → store_key mapping to prevent unbounded growth
        req_id = str(getattr(request, "request_id", ""))
        if req_id:
            _REQUEST_STORE_KEYS.pop(req_id, None)
        return (True, None)

    # ----- Load side (Agent B starts → KV injected from store) -----

    def start_load_kv(self, forward_context: Any = None, **kwargs) -> None:
        """Inject stored KV into the paged buffer before forward pass.

        Uses pending load metadata (block_ids from scheduler) to compute
        slot_mapping, then writes stored KV into the allocated blocks.
        """
        if forward_context is None or self._kv_caches is None:
            return

        # Read from module-level dict (set by scheduler instance)
        pending = dict(_PENDING_LOADS)
        if not pending:
            return

        logger.debug(
            "start_load_kv: %d pending loads, kv_caches registered",
            len(pending),
        )

        for req_id, meta in pending.items():
            try:
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

                    if layers_loaded == 0:
                        logger.debug(
                            "Injecting KV: store_key=%s, %d tokens into %d slots",
                            meta.store_key, stored_kv.shape[1],
                            slot_mapping.shape[0],
                        )

                    # Align stored KV to allocated slot count. The store may
                    # have more tokens than the scheduler allocated (the -1 cap
                    # in get_num_new_matched_tokens leaves 1 token for prefill).
                    num_slots = slot_mapping.shape[0]
                    if stored_kv.shape[1] > num_slots:
                        stored_kv = stored_kv[:, :num_slots, :]

                    _inject_request_kv(kv_cache, slot_mapping, stored_kv)
                    layers_loaded += 1

                if layers_loaded > 0:
                    logger.debug(
                        "Loaded KV for %s: %d layers, %d tokens",
                        meta.store_key, layers_loaded, meta.num_external_tokens,
                    )
            except Exception as e:
                logger.warning("start_load_kv failed for %s: %s", req_id, e)

        # Clear processed loads from module-level dict
        for req_id in pending:
            _PENDING_LOADS.pop(req_id, None)

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

        # Cap at prompt_len - 1 to leave 1 token for the scheduler.
        # This is the standard pattern for synchronous KV connectors:
        # vLLM requires at least 1 new token per forward pass to produce
        # logits for the first decode token. The ExampleConnector does the
        # same (matches prompt_token_ids[:-1]). Even the async path applies
        # this adjustment internally in _update_waiting_for_remote_kv.
        prompt_len = getattr(request, "num_tokens", None)
        if prompt_len is None:
            prompt_ids = getattr(request, "prompt_token_ids", None)
            prompt_len = len(prompt_ids) if prompt_ids else 0
        if prompt_len > 0 and matched >= prompt_len - num_computed_tokens:
            matched = max(0, prompt_len - num_computed_tokens - 1)

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

        # Use module-level dict (shared between scheduler and worker instances)
        _PENDING_LOADS[meta.request_id] = meta

        logger.debug(
            "Scheduled load for %s: %d external tokens, %d blocks",
            meta.store_key, num_external_tokens, len(block_list),
        )

    def build_connector_meta(self, scheduler_output: Any) -> "KVConnectorMetadata":
        """Build metadata to pass store keys from scheduler to worker.

        The scheduler has access to prompt_token_ids (for hashing).
        The worker needs the store key for save_kv_layer. This bridges them.
        """
        req_metas = []
        try:
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
            if new_reqs:
                for req in new_reqs:
                    meta = AVPReqMeta.from_request(req)
                    # Register request_id → store_key mapping for worker
                    _REQUEST_STORE_KEYS[meta.request_id] = meta.store_key
                    # Check if we have KV for this request (load path)
                    if self._store.has_key(meta.store_key):
                        meta.num_external_tokens = self._store.get_seq_len(meta.store_key)
                    req_metas.append(meta)
        except Exception as e:
            logger.debug("build_connector_meta: %s", e)

        return AVPConnectorMetadata(requests=req_metas)

    # ----- Registration and stats -----

    def register_kv_caches(self, kv_caches: Dict[str, Any]) -> None:
        """Store reference to vLLM's GPU KV cache buffers.

        These are used in request_finished (extraction) and
        start_load_kv (injection).
        """
        self._kv_caches = kv_caches
        logger.debug(
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


_MAX_LATENT_STEPS = 100


def prepare_latent_prompt(token_ids: List[int], latent_steps: int = 20) -> List[int]:
    """Pad prompt with N copies of the last token for extend-pattern latent thinking.

    The model plugin detects the padding and runs N latent steps at
    positions L..L+N-1, creating a causal chain of enriched KV entries.
    The placeholder KV is overwritten before being read (safe for any token).

    Args:
        token_ids: Original prompt token IDs.
        latent_steps: Number of latent thinking steps (default: 20, max: 100).

    Returns:
        Padded token IDs: original + N copies of the last token.

    Raises:
        ValueError: If latent_steps exceeds the maximum.
    """
    if latent_steps > _MAX_LATENT_STEPS:
        raise ValueError(
            f"latent_steps={latent_steps} exceeds maximum {_MAX_LATENT_STEPS}. "
            "Beyond 20 steps, accuracy degrades due to noise accumulation."
        )
    if not token_ids or latent_steps <= 0:
        return list(token_ids)
    return list(token_ids) + [token_ids[-1]] * latent_steps


def compute_request_hash(token_ids: List[int]) -> str:
    data = ",".join(str(t) for t in token_ids).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def load_projected_embedding(store_dir: str, store_key: str) -> Optional[Any]:
    """Load a projected embedding from the file store.

    Helper for Agent B to retrieve the rosetta-projected embedding
    that Agent A saved after its latent thinking steps.

    Args:
        store_dir: Path to the shared AVP store directory.
        store_key: The store key (typically from compute_request_hash).

    Returns:
        Projected embedding tensor [D_tgt], or None if not found.
    """
    store = FileKVStore(store_dir)
    return store.load_projected(store_key)


def generate_with_rosetta(
    engine: Any,
    prompt_token_ids: List[int],
    store_dir: str,
    store_key: str,
    sampling_params: Any,
    model_id: Optional[str] = None,
) -> Any:
    """Generate from a rosetta-projected embedding via vLLM.

    Loads the projected embedding that Agent A saved, prepends it to
    the prompt as a virtual context token, and generates via vLLM's
    ``prompt_embeds`` pathway. Falls back to normal token-based
    generation if no projected embedding is found.

    The engine must be created with ``enable_prompt_embeds=True``.

    Args:
        engine: vLLM ``LLM`` instance for Agent B.
        prompt_token_ids: Agent B's prompt token IDs.
        store_dir: Path to the shared AVP store directory.
        store_key: Store key from Agent A (from ``compute_request_hash``).
        sampling_params: vLLM ``SamplingParams``.
        model_id: Agent B's HuggingFace model ID. Auto-detected from
            engine if not provided.

    Returns:
        vLLM ``RequestOutput`` list (same as ``engine.generate``).
    """
    import torch

    projected = load_projected_embedding(store_dir, store_key)

    if projected is None:
        # No rosetta embedding — generate normally from tokens
        import vllm
        return engine.generate(
            [vllm.TokensPrompt(prompt_token_ids=list(prompt_token_ids))],
            sampling_params,
        )

    # Auto-detect model ID from the engine
    if model_id is None:
        try:
            model_id = engine.llm_engine.model_config.model
        except AttributeError:
            raise ValueError(
                "Cannot auto-detect model_id from engine. "
                "Pass model_id explicitly."
            )

    # Load target embed weights (cached after first call)
    from .vllm_model_plugin import _load_target_model_weights
    embed_weight, _, _ = _load_target_model_weights(model_id)

    # Convert token IDs → embeddings
    prompt_embeds = embed_weight[prompt_token_ids]  # [seq_len, D]

    # Prepend projected embedding as a virtual context token
    proj = projected.to(torch.float32)
    if proj.dim() == 1:
        proj = proj.unsqueeze(0)  # [1, D]
    combined = torch.cat([proj, prompt_embeds.to(torch.float32)], dim=0)

    # Cast to engine's dtype (bfloat16 for most modern models)
    try:
        engine_dtype = engine.llm_engine.model_config.dtype
    except AttributeError:
        engine_dtype = torch.bfloat16
    combined = combined.to(engine_dtype)

    return engine.generate(
        [{"prompt_embeds": combined}],
        sampling_params,
    )
