"""AVP latent thinking model plugin for vLLM.

Wraps vLLM model classes to add N latent forward-pass steps during prefill.
Each step extracts the last hidden state, projects it back to embedding space,
and feeds it as the next input -- building reasoning state in the KV-cache
without generating text tokens.

Registered via the ``vllm.general_plugins`` entry point so it is auto-discovered
by all vLLM processes (including workers spawned via multiprocessing).

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

            # Check for TP/PP and disable latent steps if needed
            self._check_parallelism(vllm_config)

            # Projection state (lazily initialized on first forward)
            self._projection_ready = False
            self._is_tied = True
            self._embed_weight = None
            self._w_realign = None
            self._target_norm = None

            # Skip latent steps during vLLM profiling (profile_run).
            # Profiling runs forward() with dummy inputs to measure GPU memory.
            # The latent loop on garbage inputs corrupts CUDA state.
            # Set to True after the first real forward pass succeeds.
            self._avp_profiling_done = False
            self._avp_forward_count = 0

            self._avp_initialized = True
            logger.info(
                "AVPLatentModel(%s) initialized with %d latent steps",
                base_cls.__name__, self._num_latent_steps,
            )

        def _check_parallelism(self, vllm_config: Any) -> None:
            """Disable latent steps under tensor/pipeline parallelism.

            With TP, embed_tokens.weight is sharded -- softmax projection
            through a partial vocabulary produces wrong results. With PP,
            only the first/last rank has embed_tokens/lm_head.
            """
            if vllm_config is None:
                return
            parallel_config = getattr(vllm_config, "parallel_config", None)
            if parallel_config is None:
                return
            tp = getattr(parallel_config, "tensor_parallel_size", 1)
            pp = getattr(parallel_config, "pipeline_parallel_size", 1)
            if tp > 1:
                logger.warning(
                    "Tensor parallelism (TP=%d) detected -- latent steps disabled. "
                    "Softmax projection requires full embedding table.", tp,
                )
                self._num_latent_steps = 0
            if pp > 1:
                logger.warning(
                    "Pipeline parallelism (PP=%d) detected -- latent steps disabled. "
                    "Latent loop requires embed_tokens and lm_head on same rank.", pp,
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
                        logger.warning(
                            "Cannot access embedding/lm_head weights -- latent steps disabled"
                        )
                        self._num_latent_steps = 0
                except Exception as e:
                    logger.warning(
                        "Realignment computation failed: %s -- latent steps disabled", e
                    )
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
            """Project last hidden state back to embedding space."""
            from ..realign import apply_realignment, project_to_embedding_space

            if self._is_tied:
                return project_to_embedding_space(
                    hidden_state, self._embed_weight, temperature=1.0
                )
            else:
                return apply_realignment(
                    hidden_state, self._w_realign, self._target_norm
                )

        def _should_think(self, seq_len: int) -> bool:
            """Only think during prefill (seq_len > threshold), not decode.

            Also skips latent steps during vLLM's profiling phase.
            vLLM calls forward() with dummy inputs during _initialize_kv_caches
            to measure GPU memory. Running the latent loop on garbage inputs
            corrupts CUDA state. We skip the first few forward calls to avoid
            this. After profiling completes, vLLM starts real inference.
            """
            if self._num_latent_steps <= 0:
                return False
            if not self._avp_profiling_done:
                self._avp_forward_count += 1
                # vLLM profiling typically runs 1-3 forward passes.
                # After 5 calls, assume profiling is done.
                if self._avp_forward_count > 5:
                    self._avp_profiling_done = True
                    logger.info("AVP: profiling phase complete, latent steps enabled")
                else:
                    return False
            should = seq_len > _PREFILL_SEQ_LEN_THRESHOLD
            if not should and self._avp_forward_count < 20:
                logger.info(
                    "AVP _should_think: seq_len=%d, threshold=%d, result=%s",
                    seq_len, _PREFILL_SEQ_LEN_THRESHOLD, should,
                )
            return should

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
            pattern: each step projects the last hidden state back to embedding
            space and feeds it as the next input.
            """
            import torch

            # Initial forward pass
            hidden_states = base_cls.forward(
                self,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

            # Determine sequence length for prefill detection.
            # vLLM v1 passes 1D input_ids (flattened across sequences).
            # shape[-1] gives total tokens. For prefill, this is >> 1.
            # For decode, this is the number of active sequences (1 token each).
            if input_ids is not None:
                if hasattr(input_ids, "shape"):
                    seq_len = input_ids.shape[0] if input_ids.dim() == 1 else input_ids.shape[-1]
                else:
                    seq_len = 1
            elif inputs_embeds is not None:
                seq_len = inputs_embeds.shape[-2] if hasattr(inputs_embeds, "shape") else 1
            else:
                seq_len = 1

            if not self._should_think(seq_len):
                return hidden_states

            logger.info(
                "Latent thinking triggered: seq_len=%d, input_ids.shape=%s",
                seq_len, input_ids.shape if input_ids is not None else "None",
            )

            # Lazy projection setup
            if not self._projection_ready:
                try:
                    self._setup_projection()
                except Exception as e:
                    logger.warning("Projection setup failed: %s -- skipping latent steps", e)
                    self._num_latent_steps = 0
                    return hidden_states

            if self._num_latent_steps <= 0:
                return hidden_states

            # Compute per-sequence last position for overwrite pattern.
            # In batched prefill, positions contains positions for ALL sequences
            # concatenated. We need the last position of the LAST sequence only,
            # since vLLM processes one sequence's latent steps at a time during
            # prefill. For single-sequence prefill (the common case), this is
            # simply positions[-1:].
            if positions is not None:
                last_pos = positions[-1:]
            else:
                last_pos = None

            # Latent thinking loop (overwrite pattern)
            t0 = time.monotonic()
            for step in range(self._num_latent_steps):
                # Extract last hidden state
                if hidden_states.dim() == 3:
                    last_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]
                elif hidden_states.dim() == 2:
                    last_hidden = hidden_states[-1:, :]  # [1, hidden_dim]
                else:
                    logger.warning(
                        "Unexpected hidden_states dim %d at step %d -- stopping",
                        hidden_states.dim(), step,
                    )
                    break

                # NaN safety check
                if torch.isnan(last_hidden).any():
                    logger.warning(
                        "NaN in hidden state at latent step %d/%d -- stopping early",
                        step + 1, self._num_latent_steps,
                    )
                    break

                # Project back to embedding space
                projected = self._project_hidden(last_hidden)
                projected_embed = (
                    projected.unsqueeze(1) if projected.dim() == 2 else projected
                )

                # Forward with projected embedding (overwrite pattern)
                hidden_states = base_cls.forward(
                    self,
                    input_ids=None,
                    positions=last_pos,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=projected_embed,
                    **kwargs,
                )

            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Latent thinking: %d steps in %.1fms (%.1fms/step)",
                step + 1, elapsed_ms,
                elapsed_ms / max(step + 1, 1),
            )

            return hidden_states

    _AVPLatentModel.__name__ = f"AVPLatent{base_cls.__name__}"
    _AVPLatentModel.__qualname__ = f"AVPLatent{base_cls.__name__}"
    return _AVPLatentModel


# ---------------------------------------------------------------------------
# Concrete model class (Qwen2)
# ---------------------------------------------------------------------------

# Build the class via factory if vLLM is available, otherwise use a stub
# that supports the same interface for testing.

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

        # Stub skips profiling guard (tests represent real inference)
        self._avp_profiling_done = True
        self._avp_forward_count = 0

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
        from ..realign import apply_realignment, project_to_embedding_space

        if self._is_tied:
            return project_to_embedding_space(
                hidden_state, self._embed_weight, temperature=1.0
            )
        else:
            return apply_realignment(
                hidden_state, self._w_realign, self._target_norm
            )

    def _should_think(self, seq_len: int) -> bool:
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

        last_pos = positions[-1:] if positions is not None else None

        for step in range(self._num_latent_steps):
            if hidden_states.dim() == 3:
                last_hidden = hidden_states[:, -1, :]
            elif hidden_states.dim() == 2:
                last_hidden = hidden_states[-1:, :]
            else:
                break

            if torch.isnan(last_hidden).any():
                break

            projected = self._project_hidden(last_hidden)
            projected_embed = projected.unsqueeze(1) if projected.dim() == 2 else projected

            hidden_states = self._mock_forward(
                input_ids=None,
                positions=last_pos,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=projected_embed,
                **kwargs,
            )

        return hidden_states


# Try to build the real class; fall back to stub
_real_cls = None
try:
    _real_cls = _build_qwen2_class()
except Exception:
    pass

AVPLatentQwen2ForCausalLM = _real_cls if _real_cls is not None else _AVPLatentStub
