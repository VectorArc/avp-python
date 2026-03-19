"""AVP latent thinking model plugin for vLLM.

Wraps vLLM model classes to add N latent forward-pass steps during prefill.
Each step extracts the last hidden state, projects it back to embedding space,
and feeds it as the next input — building reasoning state in the KV-cache
without generating text tokens.

Registered via the ``vllm.general_plugins`` entry point so it is auto-discovered
by all vLLM processes (including workers spawned via multiprocessing).

FRAGILE(vllm): F8 — Qwen2ForCausalLM import path, F9 — ModelRegistry API.
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LATENT_STEPS = 10
_PREFILL_SEQ_LEN_THRESHOLD = 2  # seq_len > this triggers thinking (skip decode)


def register():
    """Entry point called by vLLM's plugin system.

    Registers AVPLatentQwen2ForCausalLM as a model override. This function
    is referenced in pyproject.toml under ``[project.entry-points."vllm.general_plugins"]``.
    """
    from ._vllm_compat import HAS_QWEN2, HAS_VLLM_MODELS, ModelRegistry

    if not HAS_VLLM_MODELS:
        logger.warning("vLLM ModelRegistry not available — AVP model plugin not registered")
        return

    if not HAS_QWEN2:
        logger.warning("vLLM Qwen2ForCausalLM not available — AVP model plugin not registered")
        return

    # Register our wrapper under the same name so vLLM picks it up
    # when loading Qwen2-based models with AVP enabled
    ModelRegistry.register_model(
        "AVPLatentQwen2ForCausalLM", AVPLatentQwen2ForCausalLM
    )
    logger.info(
        "AVP latent model plugin registered: AVPLatentQwen2ForCausalLM "
        "(latent_steps=%s)",
        os.environ.get("AVP_LATENT_STEPS", str(_DEFAULT_LATENT_STEPS)),
    )


class AVPLatentQwen2ForCausalLM:
    """Qwen2 wrapper that adds latent thinking steps during prefill.

    Inherits from vLLM's Qwen2ForCausalLM at runtime (when vLLM is installed).
    In stub mode (no vLLM), this is a standalone class for testing.

    The latent loop:
    1. Run normal prefill via super().forward()
    2. Extract last hidden state from output
    3. Project back to embedding space (tied: softmax projection, untied: realignment)
    4. Feed projected embedding as inputs_embeds for next step (overwrite pattern)
    5. Repeat N times
    6. Return enriched hidden states (KV-cache grows by N entries)

    Configuration:
        AVP_LATENT_STEPS env var (default: 10)
    """

    def __init__(self, *, vllm_config=None, prefix: str = "", **kwargs):
        from ._vllm_compat import HAS_QWEN2, Qwen2ForCausalLM

        self._avp_initialized = False
        self._num_latent_steps = int(
            os.environ.get("AVP_LATENT_STEPS", str(_DEFAULT_LATENT_STEPS))
        )

        if HAS_QWEN2 and Qwen2ForCausalLM is not None:
            # Real vLLM mode — call parent __init__
            Qwen2ForCausalLM.__init__(
                self, vllm_config=vllm_config, prefix=prefix, **kwargs
            )
        else:
            # Stub mode for testing
            self.config = kwargs.get("config", None)
            self.model = kwargs.get("model", None)
            self.lm_head = kwargs.get("lm_head", None)

        # Projection state (lazily initialized on first forward)
        self._projection_ready = False
        self._is_tied = True
        self._embed_weight = None
        self._w_realign = None
        self._target_norm = None

        self._avp_initialized = True
        logger.info(
            "AVPLatentQwen2ForCausalLM initialized with %d latent steps",
            self._num_latent_steps,
        )

    def _setup_projection(self):
        """Detect tied/untied weights and cache projection state.

        Called lazily on first forward pass so model weights are available.
        """
        import torch

        from ..realign import needs_realignment

        config = self.config if hasattr(self, "config") else None
        if config is None:
            logger.warning("No model config available — defaulting to tied-weight projection")
            self._is_tied = True
            self._projection_ready = True
            return

        self._is_tied = not needs_realignment(config)

        if self._is_tied:
            # Tied models: use softmax projection through vocabulary
            embed = self._get_embed_weight()
            if embed is not None:
                self._embed_weight = embed.detach()
                # Compute target norm from embedding weights
                self._target_norm = embed.detach().to(torch.float32).norm(dim=1).mean()
            else:
                logger.warning("Cannot access embedding weights — latent steps disabled")
                self._num_latent_steps = 0
        else:
            # Untied models: use realignment matrix
            from ..realign import compute_realignment_matrix

            try:
                embed_in = self._get_embed_weight()
                lm_head_weight = self._get_lm_head_weight()
                if embed_in is not None and lm_head_weight is not None:
                    device = str(embed_in.device)
                    in_w = embed_in.detach().to(device=device, dtype=torch.float32)
                    out_w = lm_head_weight.detach().to(device=device, dtype=torch.float32)

                    # Truncate to shared vocab size (handles padding differences)
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
                        "Cannot access embedding/lm_head weights — latent steps disabled"
                    )
                    self._num_latent_steps = 0
            except Exception as e:
                logger.warning("Realignment computation failed: %s — latent steps disabled", e)
                self._num_latent_steps = 0

        self._projection_ready = True

    def _get_embed_weight(self) -> Optional[Any]:
        """Get input embedding weight tensor."""
        # vLLM models store embeddings differently than HF
        model = getattr(self, "model", None)
        if model is not None:
            embed = getattr(model, "embed_tokens", None)
            if embed is not None and hasattr(embed, "weight"):
                return embed.weight
        # Fallback: try HF-style
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

        Args:
            hidden_state: Tensor of shape [batch, hidden_dim] or [hidden_dim].

        Returns:
            Projected embedding of same shape.
        """
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
        """Determine if latent thinking should run.

        Only think during prefill (seq_len > threshold), not during decode
        (seq_len == 1). This prevents latent steps on every token generation.

        Args:
            seq_len: Number of input tokens in this forward pass.

        Returns:
            True if latent thinking should be applied.
        """
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
        """Forward pass with optional latent thinking steps.

        On prefill (seq_len > threshold), runs N additional forward passes
        using the overwrite pattern: each step projects the last hidden state
        back to embedding space and feeds it as the next input.

        Args:
            input_ids: Token IDs tensor.
            positions: Position IDs tensor.
            intermediate_tensors: For pipeline parallelism.
            inputs_embeds: Pre-computed input embeddings (used in latent steps).
            **kwargs: Passed through to parent forward().

        Returns:
            Hidden states tensor from the (possibly enriched) forward pass.
        """
        import torch

        from ._vllm_compat import HAS_QWEN2, Qwen2ForCausalLM

        # Initial forward pass
        if HAS_QWEN2 and Qwen2ForCausalLM is not None:
            hidden_states = Qwen2ForCausalLM.forward(
                self,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        elif hasattr(self, "_mock_forward"):
            # Testing path
            hidden_states = self._mock_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        else:
            raise RuntimeError("No vLLM model backend available")

        # Determine sequence length for prefill detection
        if input_ids is not None:
            seq_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else 1
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[-2] if hasattr(inputs_embeds, "shape") else 1
        else:
            seq_len = 1

        if not self._should_think(seq_len):
            return hidden_states

        # Lazy projection setup
        if not self._projection_ready:
            try:
                self._setup_projection()
            except Exception as e:
                logger.warning("Projection setup failed: %s — skipping latent steps", e)
                self._num_latent_steps = 0
                return hidden_states

        if self._num_latent_steps <= 0:
            return hidden_states

        # Latent thinking loop (overwrite pattern)
        for step in range(self._num_latent_steps):
            # Extract last hidden state
            if hidden_states.dim() == 3:
                last_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]
            elif hidden_states.dim() == 2:
                last_hidden = hidden_states[-1:, :]  # [1, hidden_dim]
            else:
                logger.warning(
                    "Unexpected hidden_states dim %d at step %d — stopping",
                    hidden_states.dim(), step,
                )
                break

            # NaN safety check
            if torch.isnan(last_hidden).any():
                logger.warning(
                    "NaN in hidden state at latent step %d/%d — stopping early",
                    step + 1, self._num_latent_steps,
                )
                break

            # Project back to embedding space
            projected = self._project_hidden(last_hidden)
            projected_embed = projected.unsqueeze(1) if projected.dim() == 2 else projected

            # Forward with projected embedding (overwrite pattern — same position)
            if HAS_QWEN2 and Qwen2ForCausalLM is not None:
                hidden_states = Qwen2ForCausalLM.forward(
                    self,
                    input_ids=None,
                    positions=positions[-1:] if positions is not None else None,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=projected_embed,
                    **kwargs,
                )
            elif hasattr(self, "_mock_forward"):
                hidden_states = self._mock_forward(
                    input_ids=None,
                    positions=positions[-1:] if positions is not None else None,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=projected_embed,
                    **kwargs,
                )

        return hidden_states


# Make AVPLatentQwen2ForCausalLM inherit from Qwen2ForCausalLM at runtime
# This must happen at import time so vLLM's model loading sees the correct MRO
def _patch_bases():
    from ._vllm_compat import HAS_QWEN2, Qwen2ForCausalLM

    if HAS_QWEN2 and Qwen2ForCausalLM is not None:
        # Dynamically add Qwen2ForCausalLM as a base class
        AVPLatentQwen2ForCausalLM.__bases__ = (Qwen2ForCausalLM,)


try:
    _patch_bases()
except Exception:
    pass  # Will work in stub mode without vLLM
