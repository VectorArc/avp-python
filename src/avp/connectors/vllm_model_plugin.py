"""AVP latent thinking model plugin for vLLM.

Wraps vLLM model classes to add N latent forward-pass steps during prefill.
Each step extracts the last hidden state, projects it back to embedding space,
and feeds it as the next input -- building reasoning state in the KV-cache
without generating text tokens.

The latent loop uses the extend pattern: the prompt is padded with N
placeholder tokens (via ``prepare_latent_prompt``). Each step writes to a
NEW position (L, L+1, ..., L+N-1), creating a causal chain where each
step attends to all prior latent positions. This matches the HuggingFace
reference implementation's ``generate_latent_steps()``.

Between steps, a new ForwardContext is set with decode-like attention metadata
(max_query_len=1), following the same pattern used by vLLM's EAGLE speculative
decoding proposer.

Registered via the ``vllm.general_plugins`` entry point so it is auto-discovered
by all vLLM processes (including workers spawned via multiprocessing).

FRAGILE(vllm): F8 -- Qwen2ForCausalLM import path, F9 -- ModelRegistry API.
Validated on vLLM 0.17.1 only.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_LATENT_STEPS = 20

# Cache for loaded target model weights: model_id → (embed_weight, tokenizer, config)
_target_weights_cache: Dict[str, Tuple[Any, Any, Any]] = {}


def register():
    """Entry point called by vLLM's plugin system.

    Registers AVP latent model wrappers for supported architectures:
    Qwen2, Llama, Mistral, Gemma. Users activate via
    ``hf_overrides={"architectures": ["AVPLatent<Arch>ForCausalLM"]}``.

    Uses string-form registration ("module:ClassName") for lazy loading to
    avoid CUDA re-initialization issues in vLLM worker processes.
    """
    from ._vllm_compat import HAS_VLLM_MODELS, ModelRegistry

    if not HAS_VLLM_MODELS:
        logger.warning("vLLM ModelRegistry not available -- AVP model plugin not registered")
        return

    _module = "avp.connectors.vllm_model_plugin"
    _models = {
        "AVPLatentQwen2ForCausalLM": "Qwen2ForCausalLM",
        "AVPLatentLlamaForCausalLM": "LlamaForCausalLM",
        "AVPLatentMistralForCausalLM": "MistralForCausalLM",
        "AVPLatentGemmaForCausalLM": "GemmaForCausalLM",
    }

    registered = []
    for avp_name, orig_name in _models.items():
        ModelRegistry.register_model(avp_name, f"{_module}:{avp_name}")
        registered.append(avp_name)

        # Optional: override the built-in class for auto-activation
        env_key = f"AVP_OVERRIDE_{orig_name.replace('ForCausalLM', '').upper()}"
        if os.environ.get(env_key, "0") == "1":
            ModelRegistry.register_model(orig_name, f"{_module}:{avp_name}")
            registered.append(f"{orig_name}(override)")

    logger.info(
        "AVP latent model plugin registered: %s (latent_steps=%s)",
        ", ".join(registered),
        os.environ.get("AVP_LATENT_STEPS", str(_DEFAULT_LATENT_STEPS)),
    )


# ---------------------------------------------------------------------------
# Target model weight loading (for cross-model rosetta)
# ---------------------------------------------------------------------------

def _load_target_model_weights(model_id: str) -> Tuple[Any, Any, Any]:
    """Load target model's embed_tokens weight, tokenizer, and config.

    Uses safetensors for minimal memory (only loads the embedding tensor).
    Results are cached in ``_target_weights_cache`` for reuse.

    Returns:
        (embed_weight, tokenizer, config) tuple.
    """
    if model_id in _target_weights_cache:
        return _target_weights_cache[model_id]

    import json

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    embed_weight = None
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        # Try single safetensors file
        try:
            path = hf_hub_download(model_id, "model.safetensors")
            with safe_open(path, framework="pt") as f:
                embed_weight = f.get_tensor("model.embed_tokens.weight")
        except Exception:
            # Try sharded model
            index_path = hf_hub_download(model_id, "model.safetensors.index.json")
            with open(index_path) as f:
                index = json.load(f)
            shard = index["weight_map"].get("model.embed_tokens.weight")
            if shard:
                shard_path = hf_hub_download(model_id, shard)
                with safe_open(shard_path, framework="pt") as f:
                    embed_weight = f.get_tensor("model.embed_tokens.weight")
    except ImportError:
        logger.warning("safetensors/huggingface_hub not available for weight loading")

    if embed_weight is None:
        raise RuntimeError(
            f"Could not load embed_tokens.weight from {model_id}. "
            "Ensure safetensors and huggingface_hub are installed."
        )

    result = (embed_weight, tokenizer, config)
    _target_weights_cache[model_id] = result
    logger.info(
        "Loaded target model weights: %s, embed shape=%s",
        model_id, list(embed_weight.shape),
    )
    return result


# ---------------------------------------------------------------------------
# Shared projection logic (used by both real and stub classes)
# ---------------------------------------------------------------------------

def _setup_projection_from_weights(embed_weight, lm_head_weight, is_tied):
    """Compute projection state from model weights.

    Returns (is_tied, embed_weight, w_realign, target_norm) or raises.
    """
    import torch

    if is_tied:
        target_norm = (
            embed_weight.detach().to(device="cpu", dtype=torch.float32)
            .norm(dim=1).mean().to(device=str(embed_weight.device))
        )
        return True, embed_weight.detach(), None, target_norm
    else:
        gpu_device = str(embed_weight.device)
        in_w = embed_weight.detach().to(device="cpu", dtype=torch.float32)
        out_w = lm_head_weight.detach().to(device="cpu", dtype=torch.float32)

        min_vocab = min(in_w.shape[0], out_w.shape[0])
        in_w = in_w[:min_vocab]
        out_w = out_w[:min_vocab]

        gram = torch.matmul(out_w.T, out_w)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(out_w.T, in_w)
        w_realign = torch.linalg.solve(gram, rhs)

        return (
            False,
            None,
            w_realign.to(device=gpu_device),
            in_w.norm(dim=1).mean().detach().to(device=gpu_device),
        )


def _project_hidden_state(hidden_state, is_tied, embed_weight, w_realign, target_norm):
    """Project hidden state back to embedding space."""
    from ..realign import apply_realignment, normalize_to_target, project_to_embedding_space

    if is_tied:
        projected = project_to_embedding_space(hidden_state, embed_weight, temperature=1.0)
        if target_norm is not None:
            projected = normalize_to_target(projected, target_norm)
    else:
        projected = apply_realignment(hidden_state, w_realign, target_norm)

    return projected


# ---------------------------------------------------------------------------
# Factory for real vLLM model class
# ---------------------------------------------------------------------------

def _make_latent_model_cls(base_cls: type) -> type:
    """Factory that creates a latent thinking wrapper for any vLLM model class.

    The returned class inherits from ``base_cls`` at definition time,
    avoiding runtime ``__bases__`` mutation.
    """

    class _AVPLatentModel(base_cls):

        def __init__(self, *, vllm_config=None, prefix: str = "", **kwargs):
            self._num_latent_steps = int(
                os.environ.get("AVP_LATENT_STEPS", str(_DEFAULT_LATENT_STEPS))
            )

            base_cls.__init__(self, vllm_config=vllm_config, prefix=prefix, **kwargs)

            self._vllm_config = vllm_config
            self._check_parallelism(vllm_config)

            self._block_size = 16
            if vllm_config is not None:
                cache_config = getattr(vllm_config, "cache_config", None)
                if cache_config is not None:
                    self._block_size = getattr(cache_config, "block_size", 16)

            self._projection_ready = False
            self._is_tied = True
            self._embed_weight = None
            self._w_realign = None
            self._target_norm = None

            # Cross-model rosetta projection state (read lazily — the KV
            # connector sets AVP_TARGET_MODEL during its __init__, which runs
            # AFTER the model is loaded in vLLM's init sequence).
            self._cross_model_ready = False
            self._avp_map = None
            self._source_lm_head_cpu = None

            logger.info(
                "AVPLatentModel(%s) initialized with %d latent steps, block_size=%d",
                base_cls.__name__, self._num_latent_steps, self._block_size,
            )

        def _check_parallelism(self, vllm_config: Any) -> None:
            """Disable latent steps under tensor/pipeline parallelism."""
            if vllm_config is None:
                return
            parallel_config = getattr(vllm_config, "parallel_config", None)
            if parallel_config is None:
                return
            tp = getattr(parallel_config, "tensor_parallel_size", 1)
            pp = getattr(parallel_config, "pipeline_parallel_size", 1)
            if tp > 1:
                logger.warning("TP=%d detected -- latent steps disabled.", tp)
                self._num_latent_steps = 0
            if pp > 1:
                logger.warning("PP=%d detected -- latent steps disabled.", pp)
                self._num_latent_steps = 0

        def _setup_projection(self):
            """Detect tied/untied weights and cache projection state."""
            from ..realign import needs_realignment

            config = self.config if hasattr(self, "config") else None
            if config is None:
                self._is_tied = True
                self._projection_ready = True
                return

            is_tied = not needs_realignment(config)
            embed = self._get_embed_weight()
            lm_head = self._get_lm_head_weight()

            if embed is None or (not is_tied and lm_head is None):
                logger.warning("Cannot access model weights -- latent steps disabled")
                self._num_latent_steps = 0
                self._projection_ready = True
                return

            try:
                self._is_tied, self._embed_weight, self._w_realign, self._target_norm = (
                    _setup_projection_from_weights(embed, lm_head, is_tied)
                )
            except Exception as e:
                logger.warning("Projection setup failed: %s -- latent steps disabled", e)
                self._num_latent_steps = 0

            self._projection_ready = True

        def _get_embed_weight(self) -> Optional[Any]:
            model = getattr(self, "model", None)
            if model is not None:
                embed = getattr(model, "embed_tokens", None)
                if embed is not None and hasattr(embed, "weight"):
                    return embed.weight
            if hasattr(self, "get_input_embeddings"):
                embed = self.get_input_embeddings()
                if embed is not None and hasattr(embed, "weight"):
                    return embed.weight
            return None

        def _get_lm_head_weight(self) -> Optional[Any]:
            lm_head = getattr(self, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight"):
                return lm_head.weight
            return None

        def _setup_cross_model(self) -> None:
            """Load target model weights and build AVPMap for rosetta projection.

            Called lazily on first prefill when ``avp_target_model`` is set.
            Loads only the target model's embed_tokens weight (~600MB) and
            tokenizer — not the full model.
            """
            self._cross_model_ready = True
            target_model_id = os.environ.get("AVP_TARGET_MODEL", "")
            if not target_model_id:
                return

            try:
                target_embed, target_tokenizer, target_config = (
                    _load_target_model_weights(target_model_id)
                )

                import torch

                # Source lm_head weight (CPU float32) for projection
                lm_head = self._get_lm_head_weight()
                if lm_head is None:
                    lm_head = self._get_embed_weight()  # tied weights
                if lm_head is None:
                    logger.warning("No lm_head weight for cross-model projection")
                    return
                self._source_lm_head_cpu = lm_head.detach().cpu().to(torch.float32)

                # Source model identity: prefer explicit AVP_SOURCE_MODEL
                # (set via avp_source_model in kv_connector_extra_config),
                # fall back to config._name_or_path.
                source_model_id = os.environ.get(
                    "AVP_SOURCE_MODEL",
                    getattr(self.config, "_name_or_path", "unknown"),
                )
                source_config_dict = (
                    self.config.to_dict() if hasattr(self.config, "to_dict") else {}
                )
                target_config_dict = (
                    target_config.to_dict() if hasattr(target_config, "to_dict") else {}
                )

                # Load source tokenizer. _name_or_path may be a local
                # directory that doesn't exist on this machine.
                from transformers import AutoTokenizer
                try:
                    source_tokenizer = AutoTokenizer.from_pretrained(source_model_id)
                except (OSError, ValueError):
                    logger.warning(
                        "Cannot load tokenizer for '%s' (may be a local path). "
                        "Cross-model projection disabled.",
                        source_model_id,
                    )
                    return

                # Build AVPMap via calibrate_from_weights
                from ..rosetta.calibrate import calibrate_from_weights
                self._avp_map = calibrate_from_weights(
                    source_model_id=source_model_id,
                    source_config_dict=source_config_dict,
                    target_model_id=target_model_id,
                    target_config_dict=target_config_dict,
                    target_embed_weight=target_embed,
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer,
                    auto_save=True,
                )
                logger.info(
                    "Cross-model projection ready: %s -> %s (%s)",
                    source_model_id, target_model_id, self._avp_map.method.value,
                )
            except Exception as e:
                logger.warning("Cross-model setup failed: %s", e, exc_info=True)
                self._avp_map = None

        def _project_cross_model(
            self, last_hiddens: Any, input_ids: Any,
            query_start_loc: Any, prefill_req_indices: list,
            num_prefill: int,
        ) -> None:
            """Project enriched hidden states via rosetta for cross-model transfer.

            Runs after the latent loop. For each prefill request, projects the
            enriched hidden state to the target model's embedding space and
            stores it in the module-level ``_PROJECTED_EMBEDDINGS`` dict for
            the KV connector to flush to disk.
            """
            if not self._cross_model_ready:
                self._setup_cross_model()
            if self._avp_map is None:
                return

            import torch
            from ..rosetta.project import (
                vocabulary_mediated_projection,
                vocab_overlap_projection,
            )
            from ..types import ProjectionMethod
            from .vllm_kv_connector import (
                _PROJECTED_EMBEDDINGS,
                _REQUEST_STORE_KEYS,
                compute_request_hash,
            )

            # Try to get request_id from forward context for _REQUEST_STORE_KEYS
            # lookup. The connector's build_connector_meta resolves explicit
            # store keys (from kv_transfer_params) and stores the mapping.
            _fwd_req_id = ""
            try:
                from vllm.forward_context import get_forward_context as _gfc
                _ctx = _gfc()
                _fwd_req_id = str(getattr(_ctx, "request_id", ""))
                if not _fwd_req_id and hasattr(_ctx, "requests") and _ctx.requests:
                    _fwd_req_id = str(getattr(_ctx.requests[0], "request_id", ""))
            except Exception:
                pass

            for i in range(num_prefill):
              try:
                req_idx = prefill_req_indices[i]
                chunk_start = int(query_start_loc[req_idx].item())
                chunk_end = int(query_start_loc[req_idx + 1].item())

                # CPU-side projection (~1ms)
                hidden_cpu = last_hiddens[i : i + 1].detach().cpu().to(torch.float32)

                if self._avp_map.method == ProjectionMethod.VOCAB_OVERLAP:
                    projected = vocab_overlap_projection(
                        hidden_cpu,
                        self._source_lm_head_cpu,
                        self._avp_map.w_map,
                        self._avp_map.src_indices,
                        target_norm=self._avp_map.target_norm,
                    )
                else:
                    projected = vocabulary_mediated_projection(
                        hidden_cpu,
                        self._source_lm_head_cpu,
                        self._avp_map.w_map,
                        target_norm=self._avp_map.target_norm,
                    )

                # Derive store key: use _REQUEST_STORE_KEYS (which resolves
                # explicit keys from kv_transfer_params), fall back to chunk hash.
                store_key = _REQUEST_STORE_KEYS.get(_fwd_req_id, "")
                if not store_key:
                    req_token_ids = input_ids[chunk_start:chunk_end].tolist()
                    store_key = compute_request_hash(req_token_ids)
                _PROJECTED_EMBEDDINGS[store_key] = projected.squeeze(0)

                logger.debug(
                    "Projected hidden state for cross-model: key=%s, shape=%s",
                    store_key, list(projected.shape),
                )
              except Exception as e:
                logger.warning(
                    "Cross-model projection failed for request %d: %s", i, e,
                )

        def _is_profiling(self) -> bool:
            """Detect vLLM's profiling phase (attn_metadata is None)."""
            try:
                from vllm.forward_context import get_forward_context
                ctx = get_forward_context()
                if getattr(ctx, "attn_metadata", None) is None:
                    return True
            except (ImportError, AssertionError, AttributeError):
                pass
            return False

        def _is_prefill(self) -> bool:
            """Detect prefill using max_query_len from attention metadata."""
            try:
                from vllm.forward_context import get_forward_context
                ctx = get_forward_context()
                attn_metadata = getattr(ctx, "attn_metadata", None)
                if attn_metadata is None:
                    return False
                if isinstance(attn_metadata, dict) and attn_metadata:
                    any_meta = next(iter(attn_metadata.values()))
                    return getattr(any_meta, "max_query_len", 1) > 1
                return getattr(attn_metadata, "max_query_len", 1) > 1
            except (ImportError, AssertionError, AttributeError, StopIteration):
                return False

        def _should_think(self) -> bool:
            if self._num_latent_steps <= 0:
                return False
            if self._is_profiling():
                return False
            return self._is_prefill()

        def _build_latent_attn_metadata(
            self, original_meta: Any, slot_mapping: Any,
            seq_lens: Any, block_table: Any,
        ) -> Any:
            """Build decode-like FlashAttentionMetadata for batched latent step.

            Each request gets 1 query token (decode-like). Supports
            arbitrary number of requests in the batch.
            """
            import torch

            try:
                from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
            except ImportError:
                return None

            num_reqs = seq_lens.shape[0]
            device = seq_lens.device

            query_start_loc = torch.arange(
                num_reqs + 1, dtype=torch.int32, device=device,
            )

            return FlashAttentionMetadata(
                num_actual_tokens=num_reqs,
                max_query_len=1,
                query_start_loc=query_start_loc,
                max_seq_len=int(seq_lens.max().item()),
                seq_lens=seq_lens.clone(),
                block_table=block_table.clone(),
                slot_mapping=slot_mapping,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )

        def _compute_slot(self, position: int, block_table: Any, req_idx: int = 0) -> int:
            """Compute physical KV-cache slot for a position."""
            block_idx = position // self._block_size
            block_number = block_table[req_idx, block_idx].item()
            return block_number * self._block_size + (position % self._block_size)

        def forward(
            self,
            input_ids: Any,
            positions: Any,
            intermediate_tensors: Optional[Any] = None,
            inputs_embeds: Optional[Any] = None,
            **kwargs,
        ) -> Any:
            """Forward pass with latent thinking steps (extend pattern).

            The prompt must be padded with N placeholder tokens (via
            ``prepare_latent_prompt``). The scheduler allocates L+N blocks.
            During the latent loop, positions L..L+N-1 get NEW KV entries
            (overwriting the placeholder KV). Each step sees all prior
            latent positions (causal chain matching HuggingFace extend).

            The placeholder KV at positions L..L+N-1 is overwritten before
            being read at each layer (reshape_and_cache_flash writes K,V
            BEFORE attention reads). Combined with incrementing seq_lens,
            the placeholder choice is irrelevant.

            Seed hidden state is extracted from position L-1 (the real
            last prompt token), NOT L+N-1 (the last placeholder).

            Output shape is preserved: [num_tokens, hidden_dim].
            """
            import torch

            hidden_states = base_cls.forward(
                self,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

            if not self._should_think():
                return hidden_states

            # Lazy projection setup
            if not self._projection_ready:
                self._setup_projection()

            N = self._num_latent_steps
            if N <= 0:
                return hidden_states

            # Get ForwardContext for metadata
            try:
                from vllm.forward_context import get_forward_context, set_forward_context
                fwd_ctx = get_forward_context()
                attn_metadata = fwd_ctx.attn_metadata
                if not isinstance(attn_metadata, dict) or not attn_metadata:
                    return hidden_states
            except Exception:
                return hidden_states

            first_layer_name = next(iter(attn_metadata))
            original_meta = attn_metadata[first_layer_name]

            query_start_loc = original_meta.query_start_loc
            seq_lens = original_meta.seq_lens
            block_table = original_meta.block_table
            num_reqs = seq_lens.shape[0]

            # Find prefill requests and their real prompt length (L = seq_len - N)
            prefill_req_indices = []
            real_last_indices = []  # position L-1 in the flat batch (seed)
            logit_indices = []      # position L+N-1 in the flat batch (output)
            real_prompt_lens = []   # L per request

            for i in range(num_reqs):
                total_len = int(seq_lens[i].item())
                if total_len <= N:
                    continue
                chunk_start = int(query_start_loc[i].item())
                chunk_end = int(query_start_loc[i + 1].item())
                query_len = chunk_end - chunk_start
                if query_len <= 1:
                    continue

                # This is a padded prefill request: total_len = L + N
                L = total_len - N

                # Validate padding: last N tokens should all be copies of
                # token at position L-1 (set by prepare_latent_prompt).
                # Without proper padding, the latent loop overwrites real
                # prompt KV entries, corrupting generation.
                if input_ids is not None and query_len == total_len:
                    pad_region = input_ids[chunk_start + L : chunk_end]
                    if pad_region.numel() == N:
                        seed_tok = input_ids[chunk_start + L - 1].item()
                        if not (pad_region == seed_tok).all():
                            logger.warning(
                                "Prompt does not appear padded by prepare_latent_prompt "
                                "(last %d tokens are not copies of token at position %d). "
                                "Skipping latent steps for this request to avoid corruption.",
                                N, L - 1,
                            )
                            continue

                # With chunked prefill, only part of the prompt may be in
                # this chunk. We need the FULL prompt's last N positions
                # to be in this chunk for extend to work.
                # The chunk covers positions (total_len - query_len)..total_len-1.
                chunk_first_pos = total_len - query_len
                if chunk_first_pos > L - 1:
                    # Position L-1 (seed) is not in this chunk — skip
                    continue

                # Chunk-relative indices
                seed_chunk_idx = chunk_start + (L - 1 - chunk_first_pos)
                logit_chunk_idx = chunk_end - 1  # last token in chunk = L+N-1

                prefill_req_indices.append(i)
                real_last_indices.append(seed_chunk_idx)
                logit_indices.append(logit_chunk_idx)
                real_prompt_lens.append(L)

            if not prefill_req_indices:
                return hidden_states

            num_prefill = len(prefill_req_indices)
            device = positions.device

            # Seed from position L-1 (real last prompt token), NOT L+N-1
            last_hiddens = hidden_states[real_last_indices, :]

            prefill_block_table = block_table[prefill_req_indices]

            # Extend loop: each step writes to a NEW position
            original_hidden = hidden_states
            t0 = time.monotonic()
            steps_completed = 0

            _debug = os.environ.get("AVP_DEBUG", "") == "1"

            for step in range(N):
                if _debug and torch.isnan(last_hiddens).any():
                    logger.warning("NaN at latent step %d/%d -- stopping", step + 1, N)
                    break

                projected = _project_hidden_state(
                    last_hiddens, self._is_tied,
                    self._embed_weight, self._w_realign, self._target_norm,
                )
                projected = projected.to(dtype=hidden_states.dtype)

                # Position for this step: L + step (different for each request)
                step_positions = torch.tensor(
                    [L + step for L in real_prompt_lens],
                    dtype=positions.dtype, device=device,
                )

                # Slot for this position
                step_slots = torch.tensor(
                    [self._compute_slot(L + step, prefill_block_table, req_idx=i)
                     for i, L in enumerate(real_prompt_lens)],
                    dtype=torch.int64, device=device,
                )

                # seq_lens grows each step: L + step + 1
                # (attend to prompt + all prior latent steps + this step)
                step_seq_lens = torch.tensor(
                    [L + step + 1 for L in real_prompt_lens],
                    dtype=seq_lens.dtype, device=device,
                )

                latent_meta = self._build_latent_attn_metadata(
                    original_meta, step_slots, step_seq_lens, prefill_block_table,
                )
                if latent_meta is None:
                    break

                per_layer_meta = {name: latent_meta for name in attn_metadata}
                per_layer_slots = {name: step_slots for name in attn_metadata}

                # Pass a dummy input_ids alongside inputs_embeds. When
                # CUDA graphs / torch.compile are active, the compiled
                # forward traces input_ids.size() which fails on None.
                # The model uses inputs_embeds when provided, so the
                # dummy IDs are ignored.
                dummy_ids = torch.zeros(
                    num_prefill, dtype=torch.long, device=device,
                )

                with set_forward_context(
                    per_layer_meta, self._vllm_config,
                    num_tokens=num_prefill, slot_mapping=per_layer_slots,
                ):
                    step_hidden = base_cls.forward(
                        self,
                        input_ids=dummy_ids,
                        positions=step_positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=projected,
                        **kwargs,
                    )

                last_hiddens = step_hidden[:num_prefill, :] if step_hidden.dim() == 2 else step_hidden
                steps_completed = step + 1

            if steps_completed > 0:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "Latent thinking (extend): %d reqs, %d steps in %.1fms (%.1fms/step)",
                    num_prefill, steps_completed, elapsed_ms,
                    elapsed_ms / steps_completed,
                )

                # Cross-model rosetta projection (if configured).
                # Read env var lazily — the KV connector sets it after model init.
                avp_target = os.environ.get("AVP_TARGET_MODEL", "")
                if avp_target:
                    self._project_cross_model(
                        last_hiddens, input_ids, query_start_loc,
                        prefill_req_indices, num_prefill,
                    )

                # Replace the logit position (L+N-1) with the enriched hidden
                original_hidden = original_hidden.clone()
                for i, idx in enumerate(logit_indices):
                    original_hidden[idx, :] = last_hiddens[i, :]

            return original_hidden

    _AVPLatentModel.__name__ = f"AVPLatent{base_cls.__name__}"
    _AVPLatentModel.__qualname__ = f"AVPLatent{base_cls.__name__}"
    return _AVPLatentModel


# ---------------------------------------------------------------------------
# Concrete model classes
# ---------------------------------------------------------------------------

def _build_qwen2_class():
    from ._vllm_compat import HAS_QWEN2, Qwen2ForCausalLM
    if HAS_QWEN2 and Qwen2ForCausalLM is not None:
        return _make_latent_model_cls(Qwen2ForCausalLM)
    return None


def _build_llama_class():
    from ._vllm_compat import HAS_LLAMA, LlamaForCausalLM
    if HAS_LLAMA and LlamaForCausalLM is not None:
        return _make_latent_model_cls(LlamaForCausalLM)
    return None


def _build_mistral_class():
    from ._vllm_compat import HAS_MISTRAL, MistralForCausalLM
    if HAS_MISTRAL and MistralForCausalLM is not None:
        return _make_latent_model_cls(MistralForCausalLM)
    return None


def _build_gemma_class():
    from ._vllm_compat import HAS_GEMMA, GemmaForCausalLM
    if HAS_GEMMA and GemmaForCausalLM is not None:
        return _make_latent_model_cls(GemmaForCausalLM)
    return None


class _AVPLatentStub:
    """Stub class for testing without vLLM."""

    _PREFILL_THRESHOLD = 2

    def __init__(self, *, vllm_config=None, prefix: str = "", **kwargs):
        self._num_latent_steps = int(
            os.environ.get("AVP_LATENT_STEPS", str(_DEFAULT_LATENT_STEPS))
        )
        self.config = kwargs.get("config", None)
        self.model = kwargs.get("model", None)
        self.lm_head = kwargs.get("lm_head", None)

        self._projection_ready = False
        self._is_tied = True
        self._embed_weight = None
        self._w_realign = None
        self._target_norm = None

        # Cross-model rosetta projection state
        self._avp_target_model = os.environ.get("AVP_TARGET_MODEL", "")
        self._cross_model_ready = False
        self._avp_map = None
        self._source_lm_head_cpu = None

    def _setup_projection(self):
        from ..realign import needs_realignment

        config = self.config
        if config is None:
            self._is_tied = True
            self._projection_ready = True
            return

        is_tied = not needs_realignment(config)
        embed = self._get_embed_weight()
        lm_head = self._get_lm_head_weight()

        if embed is None or (not is_tied and lm_head is None):
            self._num_latent_steps = 0
            self._projection_ready = True
            return

        try:
            self._is_tied, self._embed_weight, self._w_realign, self._target_norm = (
                _setup_projection_from_weights(embed, lm_head, is_tied)
            )
        except Exception:
            self._num_latent_steps = 0

        self._projection_ready = True

    def _get_embed_weight(self):
        model = getattr(self, "model", None)
        if model is not None:
            embed = getattr(model, "embed_tokens", None)
            if embed is not None and hasattr(embed, "weight"):
                return embed.weight
        return None

    def _get_lm_head_weight(self):
        lm_head = getattr(self, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head.weight
        return None

    def _project_hidden(self, hidden_state: Any) -> Any:
        """Convenience wrapper for tests."""
        return _project_hidden_state(
            hidden_state, self._is_tied,
            self._embed_weight, self._w_realign, self._target_norm,
        )

    def _project_cross_model_stub(self, hidden_state: Any) -> Optional[Any]:
        """Project hidden state via rosetta (stub version for testing).

        Unlike the real ``_project_cross_model`` which works with vLLM
        metadata, this takes a single hidden state tensor directly.
        Returns the projected embedding or None if cross-model is not set up.
        """
        if self._avp_map is None or self._source_lm_head_cpu is None:
            return None

        import torch
        from ..rosetta.project import (
            vocabulary_mediated_projection,
            vocab_overlap_projection,
        )
        from ..types import ProjectionMethod

        hidden_cpu = hidden_state.detach().cpu().to(torch.float32)
        if hidden_cpu.dim() == 1:
            hidden_cpu = hidden_cpu.unsqueeze(0)

        if self._avp_map.method == ProjectionMethod.VOCAB_OVERLAP:
            return vocab_overlap_projection(
                hidden_cpu,
                self._source_lm_head_cpu,
                self._avp_map.w_map,
                self._avp_map.src_indices,
                target_norm=self._avp_map.target_norm,
            )
        else:
            return vocabulary_mediated_projection(
                hidden_cpu,
                self._source_lm_head_cpu,
                self._avp_map.w_map,
                target_norm=self._avp_map.target_norm,
            )

    def _should_think(self, seq_len: int) -> bool:
        if self._num_latent_steps <= 0:
            return False
        return seq_len > self._PREFILL_THRESHOLD

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        import torch

        if not hasattr(self, "_mock_forward"):
            raise RuntimeError("No backend available and no _mock_forward set")

        hidden_states = self._mock_forward(
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds, **kwargs,
        )

        seq_len = 1
        if input_ids is not None and hasattr(input_ids, "shape"):
            seq_len = input_ids.shape[-1]
        elif inputs_embeds is not None and hasattr(inputs_embeds, "shape"):
            seq_len = inputs_embeds.shape[-2]

        if not self._should_think(seq_len):
            return hidden_states

        if not self._projection_ready:
            self._setup_projection()
        if self._num_latent_steps <= 0:
            return hidden_states

        original_hidden = hidden_states
        last_pos = positions[-1:] if positions is not None else None
        last_hidden = (
            hidden_states[-1:, :] if hidden_states.dim() == 2
            else hidden_states[:, -1, :]
        )

        steps_completed = 0
        for step in range(self._num_latent_steps):
            if torch.isnan(last_hidden).any():
                break

            projected = _project_hidden_state(
                last_hidden, self._is_tied,
                self._embed_weight, self._w_realign, self._target_norm,
            )
            if projected.dim() == 1:
                projected = projected.unsqueeze(0)

            step_hidden = self._mock_forward(
                input_ids=None, positions=last_pos,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=projected, **kwargs,
            )

            last_hidden = (
                step_hidden[:, -1, :] if step_hidden.dim() == 3
                else step_hidden[-1:, :]
            )
            steps_completed = step + 1

        if steps_completed > 0:
            original_hidden = original_hidden.clone()
            if original_hidden.dim() == 3:
                original_hidden[:, -1, :] = last_hidden
            else:
                original_hidden[-1:, :] = last_hidden

        return original_hidden


def _build_or_stub(builder):
    try:
        cls = builder()
        if cls is not None:
            return cls
    except Exception:
        pass
    return _AVPLatentStub


AVPLatentQwen2ForCausalLM = _build_or_stub(_build_qwen2_class)
AVPLatentLlamaForCausalLM = _build_or_stub(_build_llama_class)
AVPLatentMistralForCausalLM = _build_or_stub(_build_mistral_class)
AVPLatentGemmaForCausalLM = _build_or_stub(_build_gemma_class)
