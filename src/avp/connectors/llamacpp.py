"""AVP connector for llama.cpp via llama-cpp-python.

Enables latent thinking and cross-model rosetta on GGUF models running
on CPU or consumer GPUs. Uses llama.cpp's public C API
(``llama_get_embeddings_ith`` for hidden state extraction,
``batch.embd`` for embedding injection) via ctypes — no forks or
custom builds required.

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
import weakref
from typing import Any, List, Optional, Set

from .base import EngineConnector
from ..types import PayloadType

logger = logging.getLogger(__name__)

try:
    from ._llamacpp_compat import HAS_LLAMACPP
except ImportError:
    HAS_LLAMACPP = False


class LlamaCppConnector(EngineConnector):
    """AVP connector for llama.cpp GGUF models.

    Supports same-model latent thinking (via ``embeddings=True`` +
    ``llama_get_embeddings_ith``) and cross-model rosetta projection.
    Models run on CPU or GPU via llama-cpp-python.
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

        import llama_cpp  # noqa: F401

        self._model_path = model_path
        self._verbose = verbose
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._init_kwargs = kwargs

        # Load model with embedding=True to enable hidden state extraction
        # via llama_get_embeddings_ith (no cb_eval needed).
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
        self._n_layer = None
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
        output: PayloadType = PayloadType.AUTO,
        **kwargs: Any,
    ) -> Any:
        """Run N latent thinking steps and return an AVPContext.

        Creates a context with ``embeddings=True`` (no cb_eval needed).
        Hidden states are extracted via ``llama_get_embeddings_ith(-1)``
        after each decode. The context is kept alive so generate() can
        decode the solver prompt directly onto the enriched KV-cache.

        Args:
            prompt: The input prompt.
            steps: Number of latent thinking steps (default: 10).

        Returns:
            AVPContext containing the enriched hidden state and live context.
        """
        import ctypes

        import numpy as np
        from llama_cpp import llama_cpp as lc

        from ..context import AVPContext
        from ..realign import normalize_to_target, project_to_embedding_space

        model_ptr = self._model._model.model
        n_embd = self._n_embd

        # Create context with embeddings=True — gives us hidden states
        # via llama_get_embeddings_ith without cb_eval (which corrupts
        # the context and breaks generation).
        ctx_params = lc.llama_context_default_params()
        ctx_params.n_ctx = self._n_ctx
        ctx_params.n_batch = self._n_ctx  # Must fit full prompt in one decode
        ctx_params.embeddings = True

        think_ctx = lc.llama_new_context_with_model(model_ptr, ctx_params)
        if not think_ctx:
            logger.warning("think(): Failed to create context")
            return None

        # Apply chat template and tokenize. Without the template,
        # instruct models don't know when to start/stop generating.
        tokens = self._apply_chat_template(prompt)
        n_tokens = len(tokens)

        # Step 0: Initial forward pass on prompt tokens
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
                logger.warning("think(): initial decode failed (rc=%d)", rc)
                lc.llama_free(think_ctx)
                return None
        finally:
            lc.llama_batch_free(batch)

        # Extract hidden state via embeddings API (no cb_eval)
        hidden = self._get_embeddings(lc, think_ctx, n_embd)
        if hidden is None:
            logger.warning("think(): llama_get_embeddings_ith returned NULL")
            lc.llama_free(think_ctx)
            return None

        # Load embedding weights from GGUF for projection (cached, numpy)
        embed_weight, target_norm = self._get_embed_weight()
        if target_norm is None:
            target_norm = float(np.linalg.norm(hidden, axis=-1).mean())

        n_past = n_tokens
        steps_completed = 0

        # Steps 1..N: latent thinking loop
        for step in range(steps):
            # Project hidden state back to embedding space
            if embed_weight is not None:
                projected = project_to_embedding_space(
                    hidden, embed_weight, temperature=1.0,
                )
                projected = normalize_to_target(projected, target_norm)
            else:
                projected = normalize_to_target(hidden, target_norm)
            proj_np = np.ascontiguousarray(
                projected.squeeze(axis=0), dtype=np.float32,
            )

            # Inject via batch.embd
            emb_batch = lc.llama_batch_init(1, n_embd, 1)
            try:
                ctypes.memmove(
                    emb_batch.embd,
                    proj_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.sizeof(ctypes.c_float) * n_embd,
                )
                emb_batch.n_tokens = 1
                emb_batch.pos[0] = n_past
                emb_batch.seq_id[0][0] = 0
                emb_batch.n_seq_id[0] = 1
                emb_batch.logits[0] = 1

                rc = lc.llama_decode(think_ctx, emb_batch)
                if rc != 0:
                    logger.warning("think(): step %d decode failed (rc=%d)", step, rc)
                    break
            finally:
                lc.llama_batch_free(emb_batch)

            n_past += 1
            steps_completed += 1

            # Extract new hidden state
            new_hidden = self._get_embeddings(lc, think_ctx, n_embd)
            if new_hidden is not None:
                hidden = new_hidden

        hidden = np.ascontiguousarray(
            normalize_to_target(hidden, target_norm), dtype=np.float32,
        )

        logger.info(
            "think(): %d/%d latent steps, n_past=%d, hidden norm=%.3f",
            steps_completed, steps, n_past, float(np.linalg.norm(hidden)),
        )

        # Resolve AUTO → KV_CACHE (same-model default for llama.cpp)
        if output == PayloadType.AUTO:
            output = PayloadType.KV_CACHE

        # output=HIDDEN_STATE: return only the hidden state, free C context
        if output == PayloadType.HIDDEN_STATE:
            lc.llama_free(think_ctx)
            return AVPContext(
                past_key_values=None,
                model_hash="",
                num_steps=steps_completed,
                seq_len=n_past,
                hidden_dim=n_embd,
                num_layers=self._n_layer or 0,
                last_hidden_state=hidden,
            )

        # Default: return full context with live KV-cache
        context = AVPContext(
            past_key_values=None,
            model_hash="",
            num_steps=steps_completed,
            seq_len=n_past,
            hidden_dim=n_embd,
            num_layers=self._n_layer or 0,
            last_hidden_state=hidden,
        )
        context._llamacpp_ctx = think_ctx
        context._llamacpp_n_past = n_past

        # Register destructor so the C context is freed even if
        # generate() is never called. Detached in generate() paths
        # that free the context themselves to avoid double-free.
        context._llamacpp_finalizer = weakref.finalize(
            context, LlamaCppConnector._free_ctx, think_ctx,
        )

        return context

    def _apply_chat_template(self, prompt: str) -> list:
        """Apply the model's own chat template and tokenize.

        Reads the Jinja2 chat template from GGUF metadata and renders
        it with the prompt. This is model-agnostic — any model with a
        ``tokenizer.chat_template`` in its GGUF file works automatically
        (ChatML, Llama 3, Mistral, Gemma, Phi, etc.).

        Falls back to ChatML only if no template is found or rendering
        fails.
        """
        template_str = self._get_chat_template()

        if template_str:
            try:
                # Empty system message suppresses model-specific defaults
                # (e.g., "You are Qwen...") that add noise to the KV-cache
                # and hurt latent thinking quality.
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ]
                formatted = self._render_chat_template(
                    template_str, messages,
                )
                if formatted:
                    return self._tokenize_with_special(formatted)
            except Exception as e:
                logger.debug("Chat template rendering failed: %s", e)

        # Last resort fallback: ChatML (most common instruct format)
        formatted = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return self._tokenize_with_special(formatted)

    def _get_chat_template(self) -> Optional[str]:
        """Read the Jinja2 chat template from GGUF metadata.

        Returns None if no template is found.
        """
        if hasattr(self, "_cached_chat_template"):
            return self._cached_chat_template

        meta = self._model.metadata or {}
        template = None
        for key, val in meta.items():
            if "chat_template" in key:
                template = val
                break

        self._cached_chat_template = template
        return template

    @staticmethod
    def _render_chat_template(
        template_str: str,
        messages: list,
        add_generation_prompt: bool = True,
    ) -> Optional[str]:
        """Render a Jinja2 chat template with messages.

        Uses the same rendering approach as HuggingFace tokenizers and
        llama-cpp-python. Jinja2 is a transitive dependency of both
        torch and transformers, so it's always available.

        Args:
            template_str: Jinja2 template from GGUF metadata.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            add_generation_prompt: Append assistant turn start marker.

        Returns:
            Rendered prompt string, or None on failure.
        """
        try:
            from jinja2 import BaseLoader, Environment

            env = Environment(
                loader=BaseLoader(),
                keep_trailing_newline=True,
            )
            # Some templates call raise_exception for validation
            env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
                ValueError(msg),
            )
            tmpl = env.from_string(template_str)
            return tmpl.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                bos_token="",  # BOS handled by tokenize(add_bos=True)
                eos_token="",
            )
        except ImportError:
            logger.debug("jinja2 not available for template rendering")
            return None

    def _tokenize_with_special(self, text: str) -> list:
        """Tokenize text, recognizing special tokens like <|im_start|>.

        Tries ``special=True`` first (newer llama-cpp-python), falling
        back to without it for older versions.
        """
        try:
            return self._model.tokenize(
                text.encode("utf-8"), add_bos=True, special=True,
            )
        except TypeError:
            return self._model.tokenize(
                text.encode("utf-8"), add_bos=True,
            )

    @staticmethod
    def _get_embeddings(lc: Any, ctx: Any, n_embd: int) -> Any:
        """Extract hidden state from context via llama_get_embeddings_ith.

        Returns numpy array of shape [1, n_embd] (float32).
        """
        import ctypes

        import numpy as np

        emb_ptr = lc.llama_get_embeddings_ith(ctx, -1)
        if not emb_ptr:
            return None
        arr = (ctypes.c_float * n_embd).from_address(
            ctypes.addressof(emb_ptr.contents),
        )
        data = np.array(arr, dtype=np.float32, copy=True)
        return data.reshape(1, n_embd)  # [1, n_embd]

    def generate(
        self,
        prompt: str,
        context: Optional[Any] = None,
        source: Optional["LlamaCppConnector"] = None,
        cross_model: bool = False,
        max_tokens: int = 512,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """Generate text, optionally using latent context.

        When ``context`` has a live llama_context (from think()), the
        solver prompt is decoded directly onto the enriched KV-cache.
        The model attends to all prompt tokens + all N latent step
        positions, matching the HuggingFace connector's behavior.

        For cross-model or deserialized contexts (no live context),
        falls back to single-embedding injection via batch.embd.

        Args:
            prompt: The input prompt.
            context: AVPContext from a prior think() call.
            source: Source connector for cross-model rosetta.
            cross_model: Enable cross-model projection.
            max_tokens: Maximum tokens to generate.
            max_new_tokens: Alias for max_tokens (ABC compatibility).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text string.
        """
        # ABC compatibility: max_new_tokens takes precedence
        if max_new_tokens is not None:
            max_tokens = max_new_tokens

        # If no context, just generate normally
        if context is None:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return output["choices"][0]["text"]

        # Check for a live think context (full KV-cache path)
        think_ctx = getattr(context, "_llamacpp_ctx", None)
        if think_ctx is not None:
            return self._generate_on_think_ctx(
                prompt, context, max_tokens, temperature, top_p,
            )

        # No live context — use embedding injection (hidden state path).
        # Works for both same-model (output="hidden_state") and cross-model.
        hidden = getattr(context, "last_hidden_state", None)
        if hidden is None:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return output["choices"][0]["text"]

        if cross_model and source is not None:
            hidden = self._project_rosetta(hidden, source)

        return self._generate_with_embedding(
            prompt, hidden, max_tokens, temperature, top_p,
        )

    def _generate_on_think_ctx(
        self,
        prompt: str,
        context: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate on the live think context's enriched KV-cache.

        Decodes the solver prompt tokens directly onto the think context
        (which has all prompt + latent step KV entries), then runs
        autoregressive generation. The model attends to the full
        enriched context. Frees the think context when done.
        """
        from llama_cpp import llama_cpp as lc

        think_ctx = context._llamacpp_ctx
        n_past = context._llamacpp_n_past

        try:
            # If the solver prompt is different from the think prompt,
            # decode it onto the think context so the model can attend
            # to both. If same prompt (or empty), skip — the KV-cache
            # already has everything we need.
            n_cur = n_past

            if prompt:
                tokens = self._apply_chat_template(prompt)
                if tokens:
                    batch = lc.llama_batch_init(len(tokens), 0, 1)
                    try:
                        for i, tok in enumerate(tokens):
                            batch.token[i] = tok
                            batch.pos[i] = n_past + i
                            batch.seq_id[i][0] = 0
                            batch.n_seq_id[i] = 1
                            batch.logits[i] = 1 if i == len(tokens) - 1 else 0
                        batch.n_tokens = len(tokens)

                        rc = lc.llama_decode(think_ctx, batch)
                        if rc != 0:
                            logger.warning("generate(): prompt decode failed (rc=%d)", rc)
                            return ""
                    finally:
                        lc.llama_batch_free(batch)
                    n_cur = n_past + len(tokens)

            # Autoregressive generation with both token ID and text-based
            # stop detection. Token ID catches EOS/<|im_end|> when the model
            # emits them as special tokens. Text-based catches the common
            # case where quantized/small models hallucinate multi-turn
            # continuations (e.g., bare "Human:" without ChatML markers).
            sampler = self._make_sampler(lc, temperature, top_p)
            next_batch = lc.llama_batch_init(1, 0, 1)
            stop_tokens = self._get_stop_tokens()
            stop_strings = self._get_stop_strings()
            generated_text = ""
            stopped = False

            try:
                for _ in range(max_tokens):
                    token_id = lc.llama_sampler_sample(sampler, think_ctx, -1)
                    lc.llama_sampler_accept(sampler, token_id)

                    if token_id in stop_tokens:
                        stopped = True
                        break

                    # Detokenize incrementally for text-based stop check
                    try:
                        piece = self._model.detokenize(
                            [token_id], special=True,
                        ).decode("utf-8", errors="replace")
                    except TypeError:
                        piece = self._model.detokenize(
                            [token_id],
                        ).decode("utf-8", errors="replace")
                    generated_text += piece

                    # Check stop strings in recent text
                    if stop_strings:
                        tail = generated_text[-64:]
                        for ss in stop_strings:
                            if ss in tail:
                                # Truncate at the stop string
                                idx = generated_text.rfind(ss)
                                if idx >= 0:
                                    generated_text = generated_text[:idx]
                                stopped = True
                                break
                    if stopped:
                        break

                    next_batch.token[0] = token_id
                    next_batch.pos[0] = n_cur
                    next_batch.seq_id[0][0] = 0
                    next_batch.n_seq_id[0] = 1
                    next_batch.logits[0] = 1
                    next_batch.n_tokens = 1
                    n_cur += 1

                    rc = lc.llama_decode(think_ctx, next_batch)
                    if rc != 0:
                        break
            finally:
                lc.llama_sampler_free(sampler)
                lc.llama_batch_free(next_batch)

            logger.debug(
                "generate(): think_ctx with %d enriched positions + "
                "%d prompt tokens, generated %d chars, stopped=%s",
                n_past, n_cur - n_past, len(generated_text), stopped,
            )
            return generated_text

        finally:
            # Detach the destructor before manual free to avoid double-free
            finalizer = getattr(context, "_llamacpp_finalizer", None)
            if finalizer:
                finalizer.detach()
            lc.llama_free(think_ctx)
            context._llamacpp_ctx = None

    def _generate_with_embedding(
        self,
        prompt: str,
        embedding: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate text with a prepended embedding via batch.embd.

        Creates a dedicated context (not shared with the model's
        internal context) so concurrent calls don't corrupt state.

        Pipeline:
        1. Inject embedding as position 0 via batch.embd
        2. Decode prompt tokens at positions 1..L
        3. Autoregressive generation from position L+1
        """
        import ctypes
        import numpy as np
        from llama_cpp import llama_cpp

        model_ptr = self._model._model.model
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

        # Create a dedicated context to avoid corrupting the model's
        # internal context. Freed in the finally block below.
        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.n_ctx = self._n_ctx
        ctx_params.n_batch = self._n_ctx
        ctx = llama_cpp.llama_new_context_with_model(model_ptr, ctx_params)
        if not ctx:
            logger.warning("_generate_with_embedding: failed to create context")
            output = self._model(
                prompt, max_tokens=max_tokens, temperature=temperature,
            )
            return output["choices"][0]["text"]

        try:
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
                    logger.warning(
                        "Embedding decode failed (rc=%d), falling back to text", rc,
                    )
                    output = self._model(
                        prompt, max_tokens=max_tokens, temperature=temperature,
                    )
                    return output["choices"][0]["text"]
            finally:
                llama_cpp.llama_batch_free(emb_batch)

            # Step 2: Decode prompt tokens at positions 1..L
            tokens = self._apply_chat_template(prompt)
            n_prompt = len(tokens)

            tok_batch = llama_cpp.llama_batch_init(n_prompt, 0, 1)
            try:
                for i, tok in enumerate(tokens):
                    tok_batch.token[i] = tok
                    tok_batch.pos[i] = i + 1  # offset by 1 for embedding prefix
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

            # Step 3: Autoregressive generation with text-based stop detection
            sampler = self._make_sampler(llama_cpp, temperature, top_p)
            n_cur = 1 + n_prompt  # current position (1 embd + L prompt)
            stop_tokens = self._get_stop_tokens()
            stop_strings = self._get_stop_strings()
            generated_text = ""
            stopped = False

            next_batch = llama_cpp.llama_batch_init(1, 0, 1)
            try:
                for _ in range(max_tokens):
                    token_id = llama_cpp.llama_sampler_sample(sampler, ctx, -1)
                    llama_cpp.llama_sampler_accept(sampler, token_id)

                    if token_id in stop_tokens:
                        stopped = True
                        break

                    try:
                        piece = self._model.detokenize(
                            [token_id], special=True,
                        ).decode("utf-8", errors="replace")
                    except TypeError:
                        piece = self._model.detokenize(
                            [token_id],
                        ).decode("utf-8", errors="replace")
                    generated_text += piece

                    if stop_strings:
                        tail = generated_text[-64:]
                        for ss in stop_strings:
                            if ss in tail:
                                idx = generated_text.rfind(ss)
                                if idx >= 0:
                                    generated_text = generated_text[:idx]
                                stopped = True
                                break
                    if stopped:
                        break

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
                llama_cpp.llama_sampler_free(sampler)
                llama_cpp.llama_batch_free(next_batch)

            logger.debug(
                "generate(): injected embedding + %d prompt tokens, "
                "generated %d chars, stopped=%s",
                n_prompt, len(generated_text), stopped,
            )
            return generated_text

        finally:
            llama_cpp.llama_free(ctx)

    def _get_stop_tokens(self) -> set:
        """Get the set of token IDs that should stop generation.

        Includes EOS and chat template end markers (<|im_end|> for ChatML,
        </s> for Llama, <|eot_id|> for Llama 3). Without these, the model
        generates past the answer into fake multi-turn conversations.

        Uses safe Python-level APIs only (no direct ctypes C calls) to
        avoid segfaults across llama-cpp-python versions:
        1. GGUF metadata lookup for eot_token_id
        2. tokenize(..., special=True) on newer llama-cpp-python
        3. detokenize scan over tail of vocabulary

        Results are cached after first call.
        """
        if hasattr(self, "_cached_stop_tokens"):
            return self._cached_stop_tokens

        stops = set()

        # Always include EOS
        eos = self._model.token_eos()
        if eos >= 0:
            stops.add(eos)

        # Strategy 1: read eot_token_id from GGUF metadata (safest)
        meta = self._model.metadata or {}
        for key, val in meta.items():
            if "eot_token_id" in key:
                try:
                    eot = int(val)
                    if 0 <= eot < self._n_vocab:
                        stops.add(eot)
                        logger.debug("Found EOT from metadata %s: %d", key, eot)
                except (ValueError, TypeError):
                    pass

        # Strategy 2: derive stop markers from chat template, then tokenize
        # This is model-agnostic — works for any model with a template.
        stop_markers = self._get_stop_strings()  # template-derived
        for marker in stop_markers:
            # Only try to tokenize special-token-shaped markers
            if not (marker.startswith("<|") or marker.startswith("[/")):
                continue
            try:
                ids = self._model.tokenize(
                    marker.encode("utf-8"), add_bos=False, special=True,
                )
                if ids and len(ids) == 1:
                    stops.add(ids[0])
                    logger.debug(
                        "Found stop token via tokenize: %s = %d", marker, ids[0],
                    )
            except (TypeError, AttributeError):
                pass

        # Strategy 3: detokenize scan over tail of vocabulary
        # Only if strategies 1-2 found nothing beyond EOS.
        # Uses template-derived markers as the search target.
        if len(stops) <= 1:
            search_strings = {
                m for m in stop_markers
                if m.startswith("<|") or m.startswith("[/") or m == "</s>"
            }
            if search_strings:
                start = max(0, self._n_vocab - 1000)
                for token_id in range(start, self._n_vocab):
                    try:
                        piece = self._model.detokenize(
                            [token_id], special=True,
                        )
                    except TypeError:
                        piece = self._model.detokenize([token_id])
                    try:
                        text = piece.decode("utf-8", errors="replace")
                        if text in search_strings:
                            stops.add(token_id)
                            logger.debug(
                                "Found stop token via detokenize: %s = %d",
                                text, token_id,
                            )
                            if len(stops) > 3:
                                break
                    except Exception:
                        continue

        logger.info("Stop tokens: %s", stops)
        self._cached_stop_tokens = stops
        return stops

    def _get_stop_strings(self) -> list:
        """Get text-based stop strings for generation.

        Derives stop strings from the model's chat template by rendering
        a sample conversation and extracting the markers that appear
        between/after turns. Falls back to generic patterns if no
        template is available.

        This is model-agnostic — new model families with custom
        templates work automatically.
        """
        if hasattr(self, "_cached_stop_strings"):
            return self._cached_stop_strings

        stops = []
        template_str = self._get_chat_template()

        if template_str:
            stops = self._extract_stop_strings_from_template(template_str)

        if not stops:
            # Generic fallback when no template or extraction fails
            stops = ["<|im_end|>", "</s>"]

        # Universal stop strings: common model behaviors that can't be
        # derived from chat templates. <|endoftext|> is emitted by many
        # models (Qwen, GPT-family) as the actual stop signal even when
        # the chat template uses different markers.
        for universal in ["<|endoftext|>"]:
            if universal not in stops:
                stops.append(universal)

        # Always add multi-turn hallucination guards
        stops.extend(["\nHuman:", "\nUser:"])

        logger.info("Stop strings: %s", stops)
        self._cached_stop_strings = stops
        return stops

    @staticmethod
    def _extract_stop_strings_from_template(template_str: str) -> list:
        """Extract end-of-turn markers from a chat template.

        Renders a sample two-turn conversation with sentinel content,
        then extracts the text between the assistant's response and the
        next user turn. These markers are what the model should emit
        to signal end-of-generation.

        Returns a list of stop strings, or empty list on failure.
        """
        try:
            sentinel = "SENTINEL_RESPONSE_TEXT"
            # Render: user → assistant(sentinel) → user
            # The text between sentinel and the second user message
            # contains the end-of-turn markers.
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": sentinel},
                {"role": "user", "content": "World"},
            ]
            full = LlamaCppConnector._render_chat_template(
                template_str, messages, add_generation_prompt=False,
            )
            if not full or sentinel not in full:
                return []

            # Also render just user + generation prompt to find the
            # start-of-assistant marker (for detecting new turns)
            gen_prompt = LlamaCppConnector._render_chat_template(
                template_str,
                [{"role": "user", "content": "Hello"}],
                add_generation_prompt=True,
            )

            # Extract: everything after sentinel up to "World"
            after_sentinel = full.split(sentinel, 1)[1]
            before_next_user = after_sentinel.split("World", 1)[0]
            # This contains end-of-turn markers + next-turn prefix

            stops = []
            # Extract individual special token patterns (<|...|>, [/INST], etc.)
            import re
            special_tokens = re.findall(
                r"<\|[^|]+\|>|\[/?[A-Z]+\]|</s>", before_next_user,
            )
            for tok in special_tokens:
                if tok not in stops:
                    stops.append(tok)

            # Also add the start-of-next-turn marker if we can find it
            if gen_prompt:
                after_hello = gen_prompt.split("Hello", 1)
                if len(after_hello) > 1:
                    turn_markers = re.findall(
                        r"<\|[^|]+\|>|\[/?[A-Z]+\]|</s>",
                        after_hello[1],
                    )
                    # The last special token before generation is the
                    # start-of-assistant marker — stop if we see it mid-gen
                    for tok in turn_markers:
                        if tok not in stops:
                            stops.append(tok)

            return stops
        except Exception:
            return []

    def _get_embed_weight(self) -> tuple:
        """Get cached embedding weight matrix and target norm as numpy arrays.

        Extracts and dequantizes token_embd.weight from the GGUF file
        on first call, then returns the cached version.

        Returns:
            (embed_weight_np, target_norm_float) or (None, None) on failure.
        """
        if hasattr(self, "_embed_weight_cache"):
            return self._embed_weight_cache

        try:
            import numpy as np
            from ._llamacpp_compat import extract_gguf_embedding_weights
            emb_np = extract_gguf_embedding_weights(self._model_path)
            target_norm = float(np.linalg.norm(emb_np, axis=1).mean())
            self._embed_weight_cache = (emb_np, target_norm)
            return self._embed_weight_cache
        except Exception as e:
            logger.warning("Cannot load embed weights for projection: %s", e)
            self._embed_weight_cache = (None, None)
            return (None, None)

    @staticmethod
    def _free_ctx(ctx: Any) -> None:
        """Release a llama_context. Used as weakref.finalize callback."""
        if ctx is not None:
            try:
                from llama_cpp import llama_cpp as lc
                lc.llama_free(ctx)
            except Exception:
                pass

    @staticmethod
    def _release_ctx(context: Any) -> None:
        """Detach finalizer and free C context from an AVPContext."""
        think_ctx = getattr(context, "_llamacpp_ctx", None)
        if think_ctx is None:
            return
        finalizer = getattr(context, "_llamacpp_finalizer", None)
        if finalizer:
            finalizer.detach()
        try:
            from llama_cpp import llama_cpp as lc
            lc.llama_free(think_ctx)
        except Exception:
            pass
        context._llamacpp_ctx = None

    @staticmethod
    def _make_sampler(lc: Any, temperature: float, top_p: float) -> Any:
        """Create a sampler chain matching llama-cpp-python's behavior.

        When temperature=0, uses greedy sampling only (no top_k/top_p/dist).
        This matches llama-cpp-python's _init_sampler which switches to
        greedy for temp=0. Using temp(0)+dist(0) produces garbage due to
        -inf logit handling in the softmax.
        """
        sampler_params = lc.llama_sampler_chain_default_params()
        sampler = lc.llama_sampler_chain_init(sampler_params)

        if temperature == 0.0:
            greedy_s = lc.llama_sampler_init_greedy()
            lc.llama_sampler_chain_add(sampler, greedy_s)
        else:
            top_k_s = lc.llama_sampler_init_top_k(40)
            lc.llama_sampler_chain_add(sampler, top_k_s)

            top_p_s = lc.llama_sampler_init_top_p(top_p, 1)
            lc.llama_sampler_chain_add(sampler, top_p_s)

            temp_s = lc.llama_sampler_init_temp(temperature)
            lc.llama_sampler_chain_add(sampler, temp_s)

            dist_s = lc.llama_sampler_init_dist(0)
            lc.llama_sampler_chain_add(sampler, dist_s)

        return sampler

    def _project_rosetta(self, hidden: Any, source: "LlamaCppConnector") -> Any:
        """Project hidden state from source to target model space.

        Extracts embedding weights from both GGUF files and uses
        vocabulary-mediated projection (same algorithm as HuggingFace
        connector). Requires the ``gguf`` package for dequantization.

        Returns a numpy array.
        """
        import numpy as np
        from ..rosetta.project import vocabulary_mediated_projection

        # Use cached embed weights (numpy arrays) from both connectors
        tgt_weight, target_norm = self._get_embed_weight()
        src_weight, _ = source._get_embed_weight()

        if tgt_weight is None:
            logger.warning("Cannot load target embed weights for rosetta")
            return hidden
        if src_weight is None:
            logger.warning("Cannot load source embed weights for rosetta")
            return hidden

        projected = vocabulary_mediated_projection(
            hidden,
            src_weight,
            tgt_weight,
            temperature=1.0,
            target_norm=target_norm,
        )

        logger.info(
            "Rosetta projection: [%d] -> [%d], norm=%.3f",
            hidden.shape[-1], projected.shape[-1],
            float(np.linalg.norm(projected)),
        )
        return projected

    # --- EngineConnector ABC implementation ---

    def get_model_identity(self):
        from ..types import ModelIdentity
        return ModelIdentity(
            model_id=self._model_path,
            hidden_dim=self._n_embd,
            num_layers=self._n_layer or 0,
        )

    # --- Model introspection overrides ---

    @property
    def context_length(self) -> int:
        return self._n_ctx

    @property
    def vocab_size(self) -> int:
        return self._n_vocab

    @property
    def dtype(self) -> str:
        # llama.cpp always returns float32 embeddings to Python
        return "float32"

    @property
    def has_tokenizer(self) -> bool:
        return True

    # --- Tokenization overrides ---

    def tokenize(self, text: str) -> List[int]:
        if not HAS_LLAMACPP:
            raise ImportError("llama-cpp-python required")
        return self._model.tokenize(text.encode("utf-8"), add_bos=False)

    def detokenize(self, token_ids: List[int]) -> str:
        if not HAS_LLAMACPP:
            raise ImportError("llama-cpp-python required")
        return self._model.detokenize(token_ids, special=True).decode(
            "utf-8", errors="replace"
        )

    def apply_chat_template(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        template = self._get_chat_template()
        if template:
            rendered = self._render_chat_template(
                template, messages, add_generation_prompt=add_generation_prompt,
            )
            if rendered:
                return rendered
        # Fallback: ChatML
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant")
        return "\n".join(parts)

    @property
    def stop_token_ids(self) -> Set[int]:
        return self._get_stop_tokens()

    @property
    def stop_strings(self) -> List[str]:
        return self._get_stop_strings()

    # --- Embedding weights ---

    def get_embedding_weights(self):
        embed_weight, _target_norm = self._get_embed_weight()
        if embed_weight is None:
            return (None, None)
        # GGUF models are typically tied-weight: input == output
        return (embed_weight, embed_weight)

    # --- Low-level stubs ---

    def extract_hidden_state(self, input_ids, attention_mask=None, past_key_values=None):
        raise NotImplementedError(
            "Use think() for hidden state extraction on llama.cpp"
        )

    def inject_and_generate(self, inputs_embeds, attention_mask=None,
                            past_key_values=None, max_new_tokens=512,
                            temperature=0.7, top_p=0.95):
        raise NotImplementedError(
            "Use generate(prompt, context=) for embedding injection on llama.cpp"
        )

    def needs_realignment(self) -> bool:
        # GGUF models typically have tied weights
        return False
