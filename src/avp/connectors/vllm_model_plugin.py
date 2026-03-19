"""AVP latent thinking model plugin for vLLM.

Wraps vLLM model classes to add N latent forward-pass steps during prefill.
Each step extracts the last hidden state, projects it back to embedding space,
and feeds it as the next input -- building reasoning state in the KV-cache
without generating text tokens.

Registered via the ``vllm.general_plugins`` entry point so it is auto-discovered
by all vLLM processes (including workers spawned via multiprocessing).

The latent loop uses the overwrite pattern: each step reuses the last prefill
position's KV-cache slot. Between steps, a new ForwardContext is set with
decode-like attention metadata (max_query_len=1), following the same pattern
used by vLLM's EAGLE speculative decoding proposer.

FRAGILE(vllm): F8 -- Qwen2ForCausalLM import path, F9 -- ModelRegistry API.
"""

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LATENT_STEPS = 20
_PREFILL_SEQ_LEN_THRESHOLD = 2  # seq_len > this triggers thinking (skip decode)


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

    # Always register under custom name (activated via hf_overrides)
    ModelRegistry.register_model("AVPLatentQwen2ForCausalLM", _cls_path)

    # Optionally override built-in Qwen2 (opt-in via env var)
    override_qwen2 = os.environ.get("AVP_OVERRIDE_QWEN2", "0") == "1"
    if override_qwen2:
        ModelRegistry.register_model("Qwen2ForCausalLM", _cls_path)

    logger.info(
        "AVP latent model plugin registered (latent_steps=%s, override_qwen2=%s)",
        os.environ.get("AVP_LATENT_STEPS", str(_DEFAULT_LATENT_STEPS)),
        override_qwen2,
    )


def _make_latent_model_cls(base_cls: type) -> type:
    """Factory that creates a latent thinking wrapper for any vLLM model class.

    This avoids runtime ``__bases__`` mutation (which breaks with metaclasses
    and pickle/cloudpickle serialization). The returned class inherits from
    ``base_cls`` at definition time.

    Args:
        base_cls: A vLLM model class (e.g., Qwen2ForCausalLM).

    Returns:
        A new class that wraps ``base_cls`` with latent thinking steps.
    """

    class _AVPLatentModel(base_cls):

        def __init__(self, *, vllm_config=None, prefix: str = "", **kwargs):
            self._avp_initialized = False
            self._num_latent_steps = int(
                os.environ.get("AVP_LATENT_STEPS", str(_DEFAULT_LATENT_STEPS))
            )

            base_cls.__init__(self, vllm_config=vllm_config, prefix=prefix, **kwargs)

            # Store vllm_config for set_forward_context calls during latent steps
            self._vllm_config = vllm_config

            # Check for TP/PP and disable latent steps if needed
            self._check_parallelism(vllm_config)

            # Extract block_size from cache config for slot computation
            self._block_size = 16  # default
            if vllm_config is not None:
                cache_config = getattr(vllm_config, "cache_config", None)
                if cache_config is not None:
                    self._block_size = getattr(cache_config, "block_size", 16)

            # Projection state (lazily initialized on first forward)
            self._projection_ready = False
            self._is_tied = True
            self._embed_weight = None
            self._w_realign = None
            self._target_norm = None

            self._avp_initialized = True
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
                logger.warning(
                    "Tensor parallelism (TP=%d) detected -- latent steps disabled.", tp,
                )
                self._num_latent_steps = 0
            if pp > 1:
                logger.warning(
                    "Pipeline parallelism (PP=%d) detected -- latent steps disabled.", pp,
                )
                self._num_latent_steps = 0

        def _setup_projection(self):
            """Detect tied/untied weights and cache projection state."""
            import torch

            from ..realign import needs_realignment

            config = self.config if hasattr(self, "config") else None
            if config is None:
                logger.warning("No model config -- defaulting to tied-weight projection")
                self._is_tied = True
                self._projection_ready = True
                return

            self._is_tied = not needs_realignment(config)

            if self._is_tied:
                embed = self._get_embed_weight()
                if embed is not None:
                    self._embed_weight = embed.detach()
                    self._target_norm = embed.detach().to(torch.float32).norm(dim=1).mean()
                else:
                    logger.warning("Cannot access embedding weights -- latent steps disabled")
                    self._num_latent_steps = 0
            else:
                try:
                    embed_in = self._get_embed_weight()
                    lm_head_weight = self._get_lm_head_weight()
                    if embed_in is not None and lm_head_weight is not None:
                        device = str(embed_in.device)
                        in_w = embed_in.detach().to(device=device, dtype=torch.float32)
                        out_w = lm_head_weight.detach().to(device=device, dtype=torch.float32)
                        min_vocab = min(in_w.shape[0], out_w.shape[0])
                        in_w = in_w[:min_vocab]
                        out_w = out_w[:min_vocab]
                        gram = torch.matmul(out_w.T, out_w)
                        reg = 1e-5 * torch.eye(
                            gram.shape[0], device=gram.device, dtype=gram.dtype
                        )
                        gram = gram + reg
                        rhs = torch.matmul(out_w.T, in_w)
                        self._w_realign = torch.linalg.solve(gram, rhs)
                        self._target_norm = in_w.norm(dim=1).mean().detach()
                    else:
                        logger.warning("Cannot access embedding/lm_head weights -- disabled")
                        self._num_latent_steps = 0
                except Exception as e:
                    logger.warning("Realignment failed: %s -- latent steps disabled", e)
                    self._num_latent_steps = 0

            self._projection_ready = True

        def _get_embed_weight(self) -> Optional[Any]:
            """Get input embedding weight tensor."""
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
            """Get output (lm_head) embedding weight tensor."""
            lm_head = getattr(self, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight"):
                return lm_head.weight
            return None

        def _project_hidden(self, hidden_state: Any) -> Any:
            """Project last hidden state back to embedding space.

            After projection, normalizes to target_norm. Without this,
            softmax-weighted average has norm ~0.05 while real embeddings
            have norm ~100, causing NaN in bfloat16 attention.
            """
            from ..realign import (
                apply_realignment,
                normalize_to_target,
                project_to_embedding_space,
            )

            if self._is_tied:
                projected = project_to_embedding_space(
                    hidden_state, self._embed_weight, temperature=1.0
                )
                # Tied: project_to_embedding_space doesn't normalize
                if self._target_norm is not None:
                    projected = normalize_to_target(projected, self._target_norm)
            else:
                # Untied: apply_realignment already normalizes to target_norm
                projected = apply_realignment(
                    hidden_state, self._w_realign, self._target_norm
                )

            return projected

        def _is_profiling(self) -> bool:
            """Detect vLLM's profiling/dummy-run phase via forward context.

            During profile_run(), attn_metadata is None. Running the latent
            loop in that state changes hidden_states shape and crashes
            _dummy_run's logit_indices post-processing.
            """
            try:
                from vllm.forward_context import get_forward_context
                ctx = get_forward_context()
                if getattr(ctx, "attn_metadata", None) is None:
                    return True
            except (ImportError, AssertionError, AttributeError):
                pass
            return False

        def _is_prefill(self) -> bool:
            """Detect prefill using max_query_len from attention metadata.

            In vLLM v1:
              - Decode-only batch: max_query_len == 1
              - Prefill or mixed batch: max_query_len > 1
            """
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
            """Determine if latent thinking should run this forward pass."""
            if self._num_latent_steps <= 0:
                return False
            if self._is_profiling():
                return False
            return self._is_prefill()

        def _build_latent_attn_metadata(
            self, original_meta: Any, slot_mapping_1tok: Any
        ) -> Any:
            """Build decode-like FlashAttentionMetadata for a single-token step.

            Reuses block_table and seq_lens from the original prefill metadata,
            but sets max_query_len=1 and uses the overwrite slot_mapping.
            """
            import torch

            try:
                from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
            except ImportError:
                # If flash_attn backend not available, try to copy the original
                logger.warning("FlashAttentionMetadata not importable -- cannot build latent metadata")
                return None

            num_reqs = original_meta.seq_lens.shape[0]
            # query_start_loc for decode: [0, 1, 2, ..., num_reqs]
            query_start_loc = torch.arange(
                num_reqs + 1, dtype=torch.int32,
                device=original_meta.seq_lens.device,
            )

            # Clone tensors to avoid corrupting original metadata.
            # Flash attention may modify seq_lens in-place, and the original
            # metadata is still used by the model runner after our forward()
            # returns (e.g., for KV connector stats, scheduler updates).
            seq_lens_clone = original_meta.seq_lens.clone()
            block_table_clone = original_meta.block_table.clone()

            return FlashAttentionMetadata(
                num_actual_tokens=1,  # single token per step
                max_query_len=1,
                query_start_loc=query_start_loc,
                max_seq_len=int(seq_lens_clone.max().item()),
                seq_lens=seq_lens_clone,
                block_table=block_table_clone,
                slot_mapping=slot_mapping_1tok,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )

        def _compute_overwrite_slot(
            self, position: Any, block_table: Any, req_idx: int = 0,
        ) -> int:
            """Compute the KV-cache slot index for a given position.

            slot = block_table[req, pos // block_size] * block_size + pos % block_size
            """
            block_idx = int(position) // self._block_size
            block_number = block_table[req_idx, block_idx].item()
            return block_number * self._block_size + (int(position) % self._block_size)

        def forward(
            self,
            input_ids: Any,
            positions: Any,
            intermediate_tensors: Optional[Any] = None,
            inputs_embeds: Optional[Any] = None,
            **kwargs,
        ) -> Any:
            """Forward pass with optional latent thinking steps.

            On prefill, runs N additional forward passes using the overwrite
            pattern. Between steps, sets a new ForwardContext with decode-like
            attention metadata so the attention kernels see consistent shapes.

            CRITICAL: Output shape must match input shape. vLLM's model runner
            indexes hidden_states using logit_indices. The latent loop enriches
            the last position in-place and returns the original shape.
            """
            import torch

            # Initial forward pass (uses existing ForwardContext from model runner)
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
                try:
                    self._setup_projection()
                except Exception as e:
                    logger.warning("Projection setup failed: %s -- skipping", e)
                    self._num_latent_steps = 0
                    return hidden_states

            if self._num_latent_steps <= 0:
                return hidden_states

            # Get current ForwardContext for metadata extraction
            try:
                from vllm.forward_context import get_forward_context, set_forward_context
                fwd_ctx = get_forward_context()
                attn_metadata = fwd_ctx.attn_metadata
                if not isinstance(attn_metadata, dict) or not attn_metadata:
                    return hidden_states
            except Exception:
                return hidden_states

            # Extract original attention metadata from first layer
            first_layer_name = next(iter(attn_metadata))
            original_meta = attn_metadata[first_layer_name]

            # Overwrite position: last token's position (reuse same KV slot)
            last_position = positions[-1]

            # Compute the physical KV-cache slot for the overwrite position
            slot_idx = self._compute_overwrite_slot(
                last_position, original_meta.block_table, req_idx=0,
            )
            slot_mapping_1tok = torch.tensor(
                [slot_idx], dtype=torch.int64, device=positions.device,
            )

            # Build decode-like attention metadata for latent steps
            latent_meta = self._build_latent_attn_metadata(original_meta, slot_mapping_1tok)
            if latent_meta is None:
                return hidden_states

            # Per-layer metadata dict (all layers use the same decode-like metadata)
            per_layer_meta = {name: latent_meta for name in attn_metadata}
            # Per-layer slot_mapping dict (for KV cache update via ForwardContext)
            per_layer_slots = {name: slot_mapping_1tok for name in attn_metadata}

            num_tokens = hidden_states.shape[0]
            logger.info(
                "Latent thinking: num_tokens=%d, overwrite_pos=%d, slot=%d",
                num_tokens, int(last_position), slot_idx,
            )

            # Save original hidden states -- we'll replace only the last position
            original_hidden = hidden_states

            # Extract last hidden state to start the latent loop
            last_hidden = hidden_states[-1:, :]  # [1, hidden_dim]

            # Latent thinking loop with proper ForwardContext per step
            t0 = time.monotonic()
            steps_completed = 0
            for step in range(self._num_latent_steps):
                # NaN safety check
                if torch.isnan(last_hidden).any():
                    logger.warning(
                        "NaN at latent step %d/%d -- stopping",
                        step + 1, self._num_latent_steps,
                    )
                    break

                # Project back to embedding space + normalize
                projected = self._project_hidden(last_hidden)
                if projected.dim() == 1:
                    projected = projected.unsqueeze(0)  # [hidden_dim] -> [1, hidden_dim]
                projected = projected.to(dtype=hidden_states.dtype)

                # Forward with new ForwardContext (decode-like, single token)
                with set_forward_context(
                    per_layer_meta,
                    self._vllm_config,
                    num_tokens=1,
                    slot_mapping=per_layer_slots,
                ):
                    step_hidden = base_cls.forward(
                        self,
                        input_ids=None,
                        positions=last_position.unsqueeze(0) if last_position.dim() == 0 else last_position[:1],
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=projected,
                        **kwargs,
                    )

                # Extract last hidden for next iteration
                last_hidden = step_hidden[-1:, :] if step_hidden.dim() == 2 else step_hidden
                steps_completed = step + 1

            elapsed_ms = (time.monotonic() - t0) * 1000
            if steps_completed > 0:
                logger.info(
                    "Latent thinking: %d steps in %.1fms (%.1fms/step)",
                    steps_completed, elapsed_ms,
                    elapsed_ms / steps_completed,
                )

            # Replace last position with enriched hidden state.
            # Preserves output shape [num_tokens, hidden_dim].
            if steps_completed > 0:
                original_hidden = original_hidden.clone()
                original_hidden[-1:, :] = last_hidden

            return original_hidden

    _AVPLatentModel.__name__ = f"AVPLatent{base_cls.__name__}"
    _AVPLatentModel.__qualname__ = f"AVPLatent{base_cls.__name__}"
    return _AVPLatentModel


# ---------------------------------------------------------------------------
# Concrete model class (Qwen2)
# ---------------------------------------------------------------------------

def _build_qwen2_class():
    """Build AVPLatentQwen2ForCausalLM using the factory pattern."""
    from ._vllm_compat import HAS_QWEN2, Qwen2ForCausalLM

    if HAS_QWEN2 and Qwen2ForCausalLM is not None:
        return _make_latent_model_cls(Qwen2ForCausalLM)
    return None


class _AVPLatentStub:
    """Stub class for testing without vLLM.

    Provides the same interface as the factory-generated class but without
    any real vLLM base class. Tests inject a ``_mock_forward`` callable.
    """

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
        """Detect tied/untied weights and cache projection state."""
        import torch

        from ..realign import needs_realignment

        config = self.config if hasattr(self, "config") else None
        if config is None:
            self._is_tied = True
            self._projection_ready = True
            return

        self._is_tied = not needs_realignment(config)

        if self._is_tied:
            embed = self._get_embed_weight()
            if embed is not None:
                self._embed_weight = embed.detach()
                self._target_norm = embed.detach().to(torch.float32).norm(dim=1).mean()
            else:
                self._num_latent_steps = 0
        else:
            try:
                embed_in = self._get_embed_weight()
                lm_head_weight = self._get_lm_head_weight()
                if embed_in is not None and lm_head_weight is not None:
                    device = str(embed_in.device)
                    in_w = embed_in.detach().to(device=device, dtype=torch.float32)
                    out_w = lm_head_weight.detach().to(device=device, dtype=torch.float32)
                    min_vocab = min(in_w.shape[0], out_w.shape[0])
                    in_w = in_w[:min_vocab]
                    out_w = out_w[:min_vocab]
                    gram = torch.matmul(out_w.T, out_w)
                    reg = 1e-5 * torch.eye(
                        gram.shape[0], device=gram.device, dtype=gram.dtype
                    )
                    gram = gram + reg
                    rhs = torch.matmul(out_w.T, in_w)
                    self._w_realign = torch.linalg.solve(gram, rhs)
                    self._target_norm = in_w.norm(dim=1).mean().detach()
                else:
                    self._num_latent_steps = 0
            except Exception:
                self._num_latent_steps = 0

        self._projection_ready = True

    def _get_embed_weight(self) -> Optional[Any]:
        model = getattr(self, "model", None)
        if model is not None:
            embed = getattr(model, "embed_tokens", None)
            if embed is not None and hasattr(embed, "weight"):
                return embed.weight
        return None

    def _get_lm_head_weight(self) -> Optional[Any]:
        lm_head = getattr(self, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head.weight
        return None

    def _project_hidden(self, hidden_state: Any) -> Any:
        from ..realign import (
            apply_realignment,
            normalize_to_target,
            project_to_embedding_space,
        )

        if self._is_tied:
            projected = project_to_embedding_space(
                hidden_state, self._embed_weight, temperature=1.0
            )
            if self._target_norm is not None:
                projected = normalize_to_target(projected, self._target_norm)
        else:
            projected = apply_realignment(
                hidden_state, self._w_realign, self._target_norm
            )
        return projected

    def _should_think(self, seq_len: int) -> bool:
        """Stub version uses seq_len (no ForwardContext in test mode)."""
        if self._num_latent_steps <= 0:
            return False
        return seq_len > _PREFILL_SEQ_LEN_THRESHOLD

    def forward(
        self,
        input_ids: Any,
        positions: Any,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        import torch

        if not hasattr(self, "_mock_forward"):
            raise RuntimeError("No vLLM model backend available and no _mock_forward set")

        hidden_states = self._mock_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if input_ids is not None:
            seq_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else 1
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[-2] if hasattr(inputs_embeds, "shape") else 1
        else:
            seq_len = 1

        if not self._should_think(seq_len):
            return hidden_states

        if not self._projection_ready:
            try:
                self._setup_projection()
            except Exception:
                self._num_latent_steps = 0
                return hidden_states

        if self._num_latent_steps <= 0:
            return hidden_states

        original_hidden = hidden_states
        last_pos = positions[-1:] if positions is not None else None
        last_hidden = hidden_states[-1:, :] if hidden_states.dim() == 2 else hidden_states[:, -1, :]

        steps_completed = 0
        for step in range(self._num_latent_steps):
            if torch.isnan(last_hidden).any():
                break

            projected = self._project_hidden(last_hidden)
            if projected.dim() == 1:
                projected = projected.unsqueeze(0)

            step_hidden = self._mock_forward(
                input_ids=None,
                positions=last_pos,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=projected,
                **kwargs,
            )

            if step_hidden.dim() == 3:
                last_hidden = step_hidden[:, -1, :]
            else:
                last_hidden = step_hidden[-1:, :]
            steps_completed = step + 1

        if steps_completed > 0 and original_hidden.dim() == 3:
            original_hidden = original_hidden.clone()
            original_hidden[:, -1, :] = last_hidden
        elif steps_completed > 0 and original_hidden.dim() == 2:
            original_hidden = original_hidden.clone()
            original_hidden[-1:, :] = last_hidden

        return original_hidden


# Try to build the real class; fall back to stub
_real_cls = None
try:
    _real_cls = _build_qwen2_class()
except Exception:
    pass

AVPLatentQwen2ForCausalLM = _real_cls if _real_cls is not None else _AVPLatentStub
