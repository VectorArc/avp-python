"""AVP KV-cache connector plugin for vLLM.

Implements KVConnectorBase_V1 to intercept KV-cache save/load in vLLM's
attention pipeline. Converts between vLLM's PagedAttention format and AVP's
contiguous binary wire format.

Loaded by vLLM at runtime via:
    KVTransferConfig(
        kv_connector="AVPKVConnectorV1Dynamic",
        kv_role="kv_both",
        kv_connector_module_path="avp.connectors.vllm_kv_connector"
    )

Uses a file-based store for KV-cache exchange between agents.
Extendable to Redis/HTTP backends later.
"""

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Import vLLM base class or use stub for development on non-Linux platforms
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

    class KVConnectorRole:
        """Stub for development."""
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorBase_V1:
        """Stub base class for development on non-Linux platforms."""
        def __init__(self, **kwargs):
            pass

    class KVConnectorMetadata:
        """Stub metadata class."""
        pass


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "torch is required for AVP KV connector. Install with: pip install avp[latent]"
        )


@dataclass
class _LayerBuffer:
    """Buffers per-layer KV tensors during a single request's forward pass."""
    tensors: Dict[str, Any] = field(default_factory=dict)  # layer_name → tensor


class _AVPConnectorMetadata(KVConnectorMetadata):
    """Minimal metadata for AVP's file-based KV store."""
    pass


class AVPKVConnectorV1Dynamic(KVConnectorBase_V1):
    """AVP KV-cache connector for vLLM.

    Intercepts save_kv_layer/wait_for_layer_load calls in vLLM's attention
    pipeline to serialize/deserialize KV-cache in AVP binary format.

    KV-cache exchange uses a file-based store: each file is named by
    ``{request_hash}.avp`` in the configured store directory. The request_hash
    is derived from the prompt token IDs so that a consumer can look up
    pre-computed KV-cache from a producer.

    Configuration via environment variables:
        AVP_KV_STORE_DIR: Directory for KV-cache files (default: /tmp/avp_kv_store)
        AVP_NUM_LAYERS: Number of transformer layers (required for load)
        AVP_BLOCK_SIZE: PagedAttention block size (default: 16)
    """

    def __init__(self, vllm_config=None, role=None, **kwargs):
        if HAS_VLLM and vllm_config is not None:
            super().__init__(vllm_config=vllm_config, role=role, **kwargs)
        else:
            # Stub mode (testing without vLLM runtime)
            pass
        self._torch = _require_torch()

        # Store configuration
        self._store_dir = Path(
            os.environ.get("AVP_KV_STORE_DIR", "/tmp/avp_kv_store")
        )
        self._store_dir.mkdir(parents=True, exist_ok=True)

        self._num_layers = int(os.environ.get("AVP_NUM_LAYERS", "0"))
        self._block_size = int(os.environ.get("AVP_BLOCK_SIZE", "16"))

        # Per-request layer buffers: request_id → _LayerBuffer
        self._save_buffers: Dict[str, _LayerBuffer] = {}
        self._lock = threading.RLock()

        # Reference to vLLM's KV cache buffers (set via register_kv_caches)
        self._kv_caches: Optional[List[Any]] = None

        # Loaded KV data awaiting scatter into paged blocks
        self._loaded_kv: Dict[str, List[Tuple[Any, Any]]] = {}

        logger.info(
            "AVPKVConnectorV1Dynamic initialized: store=%s, block_size=%d",
            self._store_dir, self._block_size,
        )

    @property
    def role(self):
        return KVConnectorRole.WORKER

    # ----- Producer methods -----

    def save_kv_layer(
        self,
        layer_name: str,
        kv_tensor: Any,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        """Buffer a single layer's KV tensor during forward pass.

        Called by vLLM's attention backend for each layer during generation.

        Args:
            layer_name: Layer identifier (e.g., "model.layers.0.self_attn").
            kv_tensor: The KV tensor for this layer.
            attn_metadata: vLLM attention metadata (contains request info).
        """
        request_id = self._get_request_id(attn_metadata)

        with self._lock:
            if request_id not in self._save_buffers:
                self._save_buffers[request_id] = _LayerBuffer()
            self._save_buffers[request_id].tensors[layer_name] = kv_tensor.clone()

    def wait_for_save(self) -> None:
        """Serialize all buffered layers to AVP format and write to store.

        Called after all layers have been saved for a request.
        """
        with self._lock:
            request_ids = list(self._save_buffers.keys())

        for request_id in request_ids:
            self._flush_request(request_id)

    def request_finished(
        self,
        request: Any,
        block_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Called when a request finishes. Triggers serialization if pending.

        Args:
            request: The vLLM request object.
            block_ids: Physical block IDs used by this request.

        Returns:
            Tuple of (free_blocks, metadata). free_blocks=True allows vLLM
            to reclaim blocks.
        """
        request_id = self._get_request_id_from_request(request)

        with self._lock:
            if request_id in self._save_buffers:
                self._flush_request(request_id)

        return (True, None)

    def _flush_request(self, request_id: str) -> None:
        """Serialize buffered layers for a request and write to store."""
        from .page_convert import paged_to_contiguous
        from ..kv_cache import serialize_kv_cache

        with self._lock:
            buf = self._save_buffers.pop(request_id, None)

        if buf is None or not buf.tensors:
            return

        # Sort layers by name to ensure consistent ordering
        sorted_names = sorted(buf.tensors.keys())
        num_layers = len(sorted_names)

        # Get shape info from first layer
        first_tensor = buf.tensors[sorted_names[0]]

        # Handle two possible tensor formats:
        # 1. Already contiguous: [batch, 2, num_kv_heads, seq_len, head_dim]
        # 2. Paged: [num_blocks, 2, num_kv_heads, block_size, head_dim]
        # We store contiguous tensors directly
        legacy_kv = []
        for name in sorted_names:
            t = buf.tensors[name]
            if t.dim() == 5:
                # [batch_or_blocks, 2, num_kv_heads, tokens, head_dim]
                k = t[:1, 0:1].squeeze(1)  # [1, num_kv_heads, tokens, head_dim]
                v = t[:1, 1:2].squeeze(1)
                # If this looks like a single-sequence contiguous tensor
                if t.shape[0] == 1:
                    legacy_kv.append((k, v))
                else:
                    # Multiple blocks — need block_table context
                    # For simplicity, concatenate and assume sequential
                    num_blocks_used = t.shape[0]
                    k_cat = t[:, 0].reshape(1, t.shape[2], -1, t.shape[4])
                    v_cat = t[:, 1].reshape(1, t.shape[2], -1, t.shape[4])
                    legacy_kv.append((k_cat, v_cat))
            elif t.dim() == 4:
                # [2, num_kv_heads, seq_len, head_dim] — K and V stacked
                legacy_kv.append((t[0:1].unsqueeze(0), t[1:2].unsqueeze(0)))
            else:
                logger.warning("Unexpected tensor dim %d for layer %s", t.dim(), name)
                continue

        if not legacy_kv:
            return

        # Serialize to AVP binary format
        kv_tuple = tuple(legacy_kv)
        data, header = serialize_kv_cache(kv_tuple)

        # Write to store
        store_path = self._store_path(request_id)
        store_path.write_bytes(data)

        logger.debug(
            "Saved KV-cache for request %s: %d layers, %d bytes → %s",
            request_id, num_layers, len(data), store_path,
        )

    # ----- Consumer methods -----

    def start_load_kv(self, forward_context: Any = None, **kwargs) -> None:
        """Check store for matching KV-cache and start loading.

        Called before the forward pass to pre-load KV-cache from a producer.

        Args:
            forward_context: vLLM forward context with request information.
        """
        if forward_context is None:
            return

        request_id = self._get_request_id_from_context(forward_context)
        if not request_id:
            return

        store_path = self._store_path(request_id)
        if not store_path.exists():
            return

        from ..kv_cache import deserialize_kv_cache

        data = store_path.read_bytes()
        legacy_kv, header = deserialize_kv_cache(data)

        with self._lock:
            self._loaded_kv[request_id] = list(legacy_kv)

        logger.debug(
            "Loaded KV-cache for request %s: %d layers from %s",
            request_id, header.num_layers, store_path,
        )

    def wait_for_layer_load(
        self,
        layer_name: str,
        **kwargs,
    ) -> Optional[Any]:
        """Wait for and return a specific layer's KV data.

        Called by vLLM's attention backend to get pre-loaded KV for a layer.

        Args:
            layer_name: Layer identifier.

        Returns:
            KV tensor for the layer, or None if not available.
        """
        # Find the matching request that has loaded data
        with self._lock:
            for request_id, layers in self._loaded_kv.items():
                # Extract layer index from name (e.g., "model.layers.5.self_attn" → 5)
                layer_idx = self._parse_layer_index(layer_name)
                if layer_idx is not None and layer_idx < len(layers):
                    k, v = layers[layer_idx]
                    # Stack K and V: [2, num_kv_heads, seq_len, head_dim]
                    kv = self._torch.cat([k, v], dim=0)
                    return kv
        return None

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
        **kwargs,
    ) -> Tuple[Optional[int], bool]:
        """Check how many tokens from the store match this request.

        Args:
            request: The vLLM request object.
            num_computed_tokens: Number of tokens already computed.

        Returns:
            Tuple of (num_matched_tokens, is_async). is_async=False since
            our file-based loading is synchronous.
        """
        request_id = self._get_request_id_from_request(request)
        store_path = self._store_path(request_id)

        if not store_path.exists():
            return (0, False)

        from ..kv_cache import KVCacheHeader, _KV_HEADER_SIZE

        data = store_path.read_bytes()
        if len(data) < _KV_HEADER_SIZE:
            return (0, False)

        header = KVCacheHeader.from_bytes(data)
        matched = max(0, header.seq_len - num_computed_tokens)
        return (matched, False)

    def register_kv_caches(self, kv_caches: Dict[str, Any]) -> None:
        """Store reference to vLLM's GPU KV cache buffers.

        Called once during model initialization.

        Args:
            kv_caches: Dict mapping layer names to GPU cache tensors.
        """
        self._kv_caches = kv_caches
        logger.debug("Registered %d KV cache layers", len(kv_caches))

    # ----- Scheduler/allocation hooks -----

    def build_connector_meta(self, scheduler_output: Any) -> "KVConnectorMetadata":
        """Build connector metadata for this scheduler step.

        Returns a minimal metadata object. AVP's file-based store doesn't
        need per-step scheduler coordination.
        """
        return _AVPConnectorMetadata()

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int,
    ) -> None:
        """Update state after block allocation. No-op for file-based store."""
        pass

    # ----- Stub methods (no-op for Phase 1) -----

    def handle_preemptions(self, **kwargs) -> None:
        """Handle request preemptions (no-op)."""
        pass

    def get_block_ids_with_load_errors(self, **kwargs) -> Set[int]:
        """Return block IDs that failed to load (none for Phase 1)."""
        return set()

    def get_kv_connector_stats(self, **kwargs) -> Dict[str, Any]:
        """Return connector statistics."""
        with self._lock:
            return {
                "pending_saves": len(self._save_buffers),
                "loaded_requests": len(self._loaded_kv),
                "store_dir": str(self._store_dir),
            }

    # ----- Helpers -----

    def _store_path(self, request_id: str) -> Path:
        """Get the file path for a request's KV-cache in the store."""
        # Sanitize request_id for filesystem
        safe_id = request_id.replace("/", "_").replace("\\", "_")
        return self._store_dir / f"{safe_id}.avp"

    def _get_request_id(self, attn_metadata: Any) -> str:
        """Extract a request identifier from attention metadata."""
        if hasattr(attn_metadata, "request_id"):
            return str(attn_metadata.request_id)
        if hasattr(attn_metadata, "seq_group_metadata_list"):
            # vLLM v1: first sequence group
            groups = attn_metadata.seq_group_metadata_list
            if groups:
                return str(groups[0].request_id)
        return "default"

    def _get_request_id_from_request(self, request: Any) -> str:
        """Extract request ID from a vLLM request object."""
        if hasattr(request, "request_id"):
            return str(request.request_id)
        if isinstance(request, str):
            return request
        return "default"

    def _get_request_id_from_context(self, context: Any) -> str:
        """Extract request ID from forward context."""
        if hasattr(context, "request_id"):
            return str(context.request_id)
        if hasattr(context, "requests") and context.requests:
            return str(context.requests[0].request_id)
        return ""

    @staticmethod
    def _parse_layer_index(layer_name: str) -> Optional[int]:
        """Extract numeric layer index from a layer name.

        E.g., "model.layers.5.self_attn" → 5
        """
        import re
        match = re.search(r"layers\.(\d+)", layer_name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
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
