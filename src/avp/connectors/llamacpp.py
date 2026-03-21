"""AVP connector for llama.cpp via llama-cpp-python.

Enables latent thinking and cross-model rosetta on GGUF models running
on CPU or consumer GPUs. Uses llama.cpp's public C API (cb_eval callback
for hidden state extraction, batch.embd for embedding injection) via
ctypes — no forks or custom builds required.

Usage::

    from avp.connectors.llamacpp import LlamaCppConnector

    connector = LlamaCppConnector.from_pretrained(
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )
    context = connector.think("Analyze this problem", steps=10)
    answer = connector.generate("Solve step by step", context=context)

Cross-model::

    researcher = LlamaCppConnector.from_pretrained("qwen2-7b.gguf")
    solver = LlamaCppConnector.from_pretrained("llama3-3b.gguf")
    context = researcher.think("Analyze this", steps=10)
    answer = solver.generate("Solve it", context=context, source=researcher,
                             cross_model=True)

Requires: ``pip install avp[llamacpp]`` (installs llama-cpp-python)
"""

import logging
from typing import Any, Optional

from .base import EngineConnector

logger = logging.getLogger(__name__)

try:
    from ._llamacpp_compat import HAS_LLAMACPP
except ImportError:
    HAS_LLAMACPP = False


class LlamaCppConnector(EngineConnector):
    """AVP connector for llama.cpp GGUF models.

    Supports same-model latent thinking (via cb_eval hidden state
    extraction) and cross-model rosetta projection. Models run on
    CPU or GPU via llama-cpp-python.
    """

    can_think = True

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **kwargs: Any,
    ):
        if not HAS_LLAMACPP:
            raise ImportError(
                "LlamaCppConnector requires llama-cpp-python. "
                "Install with: pip install avp[llamacpp]"
            )

        import llama_cpp

        from ._llamacpp_compat import make_eval_callback

        self._model_path = model_path
        self._verbose = verbose
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._init_kwargs = kwargs

        # Capture dict for hidden states (populated by cb_eval callback)
        self._capture: dict = {}

        # Create callback for last-layer hidden state extraction.
        # target_layer=-1 means "capture all l_out-*, keep the last one"
        self._callback = make_eval_callback(
            target_layer=-1, capture_dict=self._capture,
        )
        # Keep a reference to prevent garbage collection
        self._callback_ref = self._callback

        # Load model — used for generate() and text-only operations
        self._model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            embedding=True,
            **kwargs,
        )

        # Extract model identity
        self._n_embd = self._model.n_embd()
        self._n_vocab = self._model.n_vocab()
        self._n_layer = getattr(
            self._model.metadata or {},
            "llama.block_count",
            None,
        )
        if self._n_layer is None:
            # Try to detect from metadata
            meta = self._model.metadata or {}
            for key, val in meta.items():
                if "block_count" in key:
                    try:
                        self._n_layer = int(val)
                    except (ValueError, TypeError):
                        pass
                    break

        logger.info(
            "LlamaCppConnector loaded: %s (n_embd=%d, n_vocab=%d, n_layer=%s)",
            model_path, self._n_embd, self._n_vocab, self._n_layer,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "LlamaCppConnector":
        """Load a GGUF model for latent communication.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size (default: 4096).
            n_gpu_layers: Number of layers to offload to GPU (-1 = all).
            verbose: Enable llama.cpp logging.

        Returns:
            A LlamaCppConnector instance.
        """
        return cls(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            **kwargs,
        )

    def think(
        self,
        prompt: str,
        steps: int = 10,
        **kwargs: Any,
    ) -> Any:
        """Run latent thinking steps and return an AVPContext.

        Tokenizes the prompt, runs a forward pass with the cb_eval
        callback active to capture the last-layer hidden state, then
        wraps it in an AVPContext for transfer to generate().

        Args:
            prompt: The input prompt.
            steps: Number of latent thinking steps (currently 1 forward
                pass — multi-step latent loop planned).

        Returns:
            AVPContext containing the captured hidden state.
        """
        from ..context import AVPContext
        from llama_cpp import llama_cpp as lc

        # Clear previous capture
        self._capture.clear()

        # Get the raw model pointer from the Llama instance
        model_ptr = self._model._model.model

        # Create a dedicated context with cb_eval set at creation time.
        # The callback MUST be in context_params BEFORE llama_new_context_with_model
        # is called — setting it after has no effect (llama.cpp copies the params).
        ctx_params = lc.llama_context_default_params()
        ctx_params.n_ctx = self._n_ctx
        ctx_params.n_batch = 512
        ctx_params.cb_eval = self._callback
        ctx_params.cb_eval_user_data = None

        think_ctx = lc.llama_new_context_with_model(model_ptr, ctx_params)
        if not think_ctx:
            logger.warning("think(): Failed to create context with cb_eval")
            return None

        try:
            # Tokenize
            tokens = self._model.tokenize(prompt.encode("utf-8"), add_bos=True)
            n_tokens = len(tokens)

            # Create batch and decode
            batch = lc.llama_batch_init(n_tokens, 0, 1)
            try:
                for i, tok in enumerate(tokens):
                    batch.token[i] = tok
                    batch.pos[i] = i
                    batch.seq_id[i][0] = 0
                    batch.n_seq_id[i] = 1
                    batch.logits[i] = 1 if i == n_tokens - 1 else 0
                batch.n_tokens = n_tokens

                rc = lc.llama_decode(think_ctx, batch)
                if rc != 0:
                    logger.warning("think(): decode failed (rc=%d)", rc)
                    return None
            finally:
                lc.llama_batch_free(batch)
        finally:
            lc.llama_free(think_ctx)

        if "data" not in self._capture:
            logger.warning(
                "think(): No hidden state captured. cb_eval callback "
                "may not have matched any l_out-* tensors."
            )
            return None

        # Build AVPContext from captured hidden state
        import torch

        hidden = torch.from_numpy(self._capture["data"].copy()).float()
        # Take the last token's hidden state
        if hidden.dim() == 2:
            hidden = hidden[-1:, :]  # [1, n_embd]

        context = AVPContext(
            past_key_values=None,
            model_hash="",
            num_steps=steps,
            seq_len=n_tokens,
            hidden_dim=self._n_embd,
            num_layers=self._n_layer or 0,
            last_hidden_state=hidden,
        )

        logger.info(
            "think(): captured layer %s, shape=%s, norm=%.3f",
            self._capture.get("layer"),
            self._capture.get("shape"),
            hidden.float().norm().item(),
        )

        return context

    def generate(
        self,
        prompt: str,
        context: Optional[Any] = None,
        source: Optional["LlamaCppConnector"] = None,
        cross_model: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """Generate text, optionally using latent context.

        When ``context`` is provided, the hidden state is injected via
        llama.cpp's ``batch.embd`` as a prefix embedding before the
        prompt tokens. The model attends to this virtual context token
        during generation.

        Args:
            prompt: The input prompt.
            context: AVPContext from a prior think() call.
            source: Source connector for cross-model rosetta.
            cross_model: Enable cross-model projection.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text string.
        """
        # If no context, just generate normally
        if context is None:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return output["choices"][0]["text"]

        # With context: inject hidden state via batch.embd
        hidden = getattr(context, "last_hidden_state", None)
        if hidden is None:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return output["choices"][0]["text"]

        # Cross-model rosetta projection
        if cross_model and source is not None:
            hidden = self._project_rosetta(hidden, source)

        return self._generate_with_embedding(
            prompt, hidden, max_tokens, temperature, top_p,
        )

    def _generate_with_embedding(
        self,
        prompt: str,
        embedding: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate text with a prepended embedding via batch.embd.

        Pipeline:
        1. Inject embedding as position 0 via batch.embd
        2. Decode prompt tokens at positions 1..L
        3. Autoregressive generation from position L+1
        """
        import ctypes
        import numpy as np
        from llama_cpp import llama_cpp

        ctx = self._model._ctx.ctx
        n_embd = self._n_embd

        # Ensure embedding is float32 numpy [n_embd]
        if hasattr(embedding, "numpy"):
            emb_np = embedding.detach().cpu().float().numpy()
        else:
            emb_np = np.asarray(embedding, dtype=np.float32)
        emb_np = emb_np.reshape(-1).astype(np.float32)
        if emb_np.shape[0] != n_embd:
            logger.warning(
                "Embedding dim %d != model n_embd %d, skipping injection",
                emb_np.shape[0], n_embd,
            )
            output = self._model(
                prompt, max_tokens=max_tokens, temperature=temperature,
            )
            return output["choices"][0]["text"]

        # Clear KV cache for a fresh context
        llama_cpp.llama_kv_self_clear(ctx)

        # Step 1: Inject embedding at position 0 via batch.embd
        emb_batch = llama_cpp.llama_batch_init(1, n_embd, 1)
        try:
            ctypes.memmove(
                emb_batch.embd,
                emb_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.sizeof(ctypes.c_float) * n_embd,
            )
            emb_batch.n_tokens = 1
            emb_batch.pos[0] = 0
            emb_batch.seq_id[0][0] = 0
            emb_batch.n_seq_id[0] = 1
            emb_batch.logits[0] = 0  # No logits needed for prefix

            rc = llama_cpp.llama_decode(ctx, emb_batch)
            if rc != 0:
                logger.warning("Embedding decode failed (rc=%d), falling back to text", rc)
                output = self._model(
                    prompt, max_tokens=max_tokens, temperature=temperature,
                )
                return output["choices"][0]["text"]
        finally:
            llama_cpp.llama_batch_free(emb_batch)

        # Step 2: Decode prompt tokens at positions 1..L
        tokens = self._model.tokenize(prompt.encode("utf-8"), add_bos=True)
        n_prompt = len(tokens)

        tok_batch = llama_cpp.llama_batch_init(n_prompt, 0, 1)
        try:
            for i, tok in enumerate(tokens):
                tok_batch.token[i] = tok
                tok_batch.pos[i] = i + 1  # offset by 1 for the embedding prefix
                tok_batch.seq_id[i][0] = 0
                tok_batch.n_seq_id[i] = 1
                tok_batch.logits[i] = 1 if i == n_prompt - 1 else 0
            tok_batch.n_tokens = n_prompt

            rc = llama_cpp.llama_decode(ctx, tok_batch)
            if rc != 0:
                logger.warning("Prompt decode failed (rc=%d)", rc)
                return ""
        finally:
            llama_cpp.llama_batch_free(tok_batch)

        # Step 3: Autoregressive generation
        # Set up sampler
        sampler_params = llama_cpp.llama_sampler_chain_default_params()
        sampler = llama_cpp.llama_sampler_chain_init(sampler_params)

        # Add sampling steps: top_k → top_p → temperature → dist
        top_k_s = llama_cpp.llama_sampler_init_top_k(40)
        llama_cpp.llama_sampler_chain_add(sampler, top_k_s)

        top_p_s = llama_cpp.llama_sampler_init_top_p(top_p, 1)
        llama_cpp.llama_sampler_chain_add(sampler, top_p_s)

        temp_s = llama_cpp.llama_sampler_init_temp(temperature)
        llama_cpp.llama_sampler_chain_add(sampler, temp_s)

        dist_s = llama_cpp.llama_sampler_init_dist(0)
        llama_cpp.llama_sampler_chain_add(sampler, dist_s)

        generated_tokens = []
        n_cur = 1 + n_prompt  # current position (1 embd + L prompt)
        eos_token = self._model.token_eos()

        try:
            for _ in range(max_tokens):
                # Sample from last position
                token_id = llama_cpp.llama_sampler_sample(sampler, ctx, -1)
                llama_cpp.llama_sampler_accept(sampler, token_id)

                if token_id == eos_token:
                    break

                generated_tokens.append(token_id)

                # Decode next token
                next_batch = llama_cpp.llama_batch_init(1, 0, 1)
                try:
                    next_batch.token[0] = token_id
                    next_batch.pos[0] = n_cur
                    next_batch.seq_id[0][0] = 0
                    next_batch.n_seq_id[0] = 1
                    next_batch.logits[0] = 1
                    next_batch.n_tokens = 1
                    n_cur += 1

                    rc = llama_cpp.llama_decode(ctx, next_batch)
                    if rc != 0:
                        break
                finally:
                    llama_cpp.llama_batch_free(next_batch)
        finally:
            llama_cpp.llama_sampler_free(sampler)

        # Detokenize
        text = self._model.detokenize(generated_tokens).decode("utf-8", errors="replace")

        logger.debug(
            "generate(): injected embedding + %d prompt tokens, generated %d tokens",
            n_prompt, len(generated_tokens),
        )
        return text

    def _project_rosetta(self, hidden: Any, source: "LlamaCppConnector") -> Any:
        """Project hidden state from source model to target model space.

        Currently experimental — full rosetta projection requires
        source/target tokenizers and embed weights from GGUF metadata.
        """
        logger.warning(
            "Cross-model rosetta for llama.cpp is experimental. "
            "Use HuggingFace connector for validated cross-model projection."
        )
        return hidden

    # --- EngineConnector ABC implementation ---

    def get_model_identity(self) -> Any:
        from ..types import ModelIdentity
        return ModelIdentity(
            model_id=self._model_path,
            hidden_size=self._n_embd,
            num_layers=self._n_layer or 0,
            vocab_size=self._n_vocab,
        )

    def extract_hidden_state(self, input_ids, attention_mask=None, past_key_values=None):
        # Hidden state extraction is done via cb_eval in think()
        raise NotImplementedError(
            "Use think() for hidden state extraction on llama.cpp"
        )

    def inject_and_generate(self, inputs_embeds, attention_mask=None,
                            past_key_values=None, max_new_tokens=256,
                            temperature=0.7, top_p=0.95):
        # Embedding injection via batch.embd planned for next iteration
        raise NotImplementedError(
            "Embedding injection via batch.embd is planned for a future release"
        )

    def get_embedding_weights(self):
        # GGUF models store weights in quantized format — extracting
        # full-precision embed weights requires dequantization
        return (None, None)

    def tokenize(self, text):
        if not HAS_LLAMACPP:
            raise ImportError("llama-cpp-python required")
        return self._model.tokenize(text.encode("utf-8"), add_bos=True)

    def needs_realignment(self) -> bool:
        # GGUF models typically have tied weights
        return False
