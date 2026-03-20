"""AVP latent thinking model plugin for vLLM.

Wraps vLLM model classes to add N latent forward-pass steps during prefill.
Each step extracts the last hidden state, projects it back to embedding space,
and feeds it as the next input -- building reasoning state in the KV-cache
without generating text tokens.

The latent loop uses the overwrite pattern: each step reuses the last prefill
position's KV-cache slot. The enriched KV entry at this position persists
through all decode tokens. Between steps, information flows through the
single overwrite position (each step reads the previous step's KV).

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
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LATENT_STEPS = 20


def register():
    """Entry point called by vLLM's plugin system.

    Registers AVPLatentQwen2ForCausalLM in two ways:

    1. Always registers under ``"AVPLatentQwen2ForCausalLM"`` so users can
       activate via ``hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]}``.

    2. If ``AVP_OVERRIDE_QWEN2=1``, also overrides the built-in
       ``"Qwen2ForCausalLM"`` entry so all Qwen2 models automatically use
       latent thinking without any ``hf_overrides``.

    Uses string-form registration ("module:ClassName") for lazy loading to
    avoid CUDA re-initialization issues in vLLM worker processes.
    """
    from ._vllm_compat import HAS_VLLM_MODELS, ModelRegistry

    if not HAS_VLLM_MODELS:
        logger.warning("vLLM ModelRegistry not available -- AVP model plugin not registered")
        return

    _cls_path = "avp.connectors.vllm_model_plugin:AVPLatentQwen2ForCausalLM"

    ModelRegistry.register_model("AVPLatentQwen2ForCausalLM", _cls_path)

    override_qwen2 = os.environ.get("AVP_OVERRIDE_QWEN2", "0") == "1"
    if override_qwen2:
        ModelRegistry.register_model("Qwen2ForCausalLM", _cls_path)

    logger.info(
        "AVP latent model plugin registered (latent_steps=%s, override_qwen2=%s)",
        os.environ.get("AVP_LATENT_STEPS", str(_DEFAULT_LATENT_STEPS)),
        override_qwen2,
    )


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
                query_len = int(query_start_loc[i + 1].item()) - int(query_start_loc[i].item())
                if query_len <= 1:
                    continue
                # This is a padded prefill request: total_len = L + N
                L = total_len - N
                batch_offset = int(query_start_loc[i].item())
                prefill_req_indices.append(i)
                real_last_indices.append(batch_offset + L - 1)  # position L-1
                logit_indices.append(batch_offset + total_len - 1)  # position L+N-1
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

            for step in range(N):
                if torch.isnan(last_hiddens).any():
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

                with set_forward_context(
                    per_layer_meta, self._vllm_config,
                    num_tokens=num_prefill, slot_mapping=per_layer_slots,
                ):
                    step_hidden = base_cls.forward(
                        self,
                        input_ids=None,
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
                # Replace the logit position (L+N-1) with the enriched hidden
                original_hidden = original_hidden.clone()
                for i, idx in enumerate(logit_indices):
                    original_hidden[idx, :] = last_hiddens[i, :]

            return original_hidden

    _AVPLatentModel.__name__ = f"AVPLatent{base_cls.__name__}"
    _AVPLatentModel.__qualname__ = f"AVPLatent{base_cls.__name__}"
    return _AVPLatentModel


# ---------------------------------------------------------------------------
# Concrete model class (Qwen2)
# ---------------------------------------------------------------------------

def _build_qwen2_class():
    from ._vllm_compat import HAS_QWEN2, Qwen2ForCausalLM
    if HAS_QWEN2 and Qwen2ForCausalLM is not None:
        return _make_latent_model_cls(Qwen2ForCausalLM)
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


_real_cls = None
try:
    _real_cls = _build_qwen2_class()
except Exception:
    pass

AVPLatentQwen2ForCausalLM = _real_cls if _real_cls is not None else _AVPLatentStub
