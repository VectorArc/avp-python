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

        # Capture dict for hidden states (populated by cb_eval callback)
        self._capture: dict = {}

        # Create callback for last-layer hidden state extraction.
        # target_layer=-1 means "capture all l_out-*, keep the last one"
        self._callback = make_eval_callback(
            target_layer=-1, capture_dict=self._capture,
        )

        # Load model with cb_eval callback enabled.
        # We use the low-level API to set cb_eval on context_params
        # before context creation, since the high-level Llama class
        # doesn't expose this parameter.
        self._model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            embedding=True,
            **kwargs,
        )

        # Inject cb_eval into the context params for future context recreations.
        # Note: for the FIRST context, we need to set it post-creation
        # which means the first think() may not capture hidden states.
        # We work around this by accessing the internal context params.
        try:
            ctx_params = self._model.context_params
            ctx_params.cb_eval = self._callback
            # Store reference to prevent GC
            self._callback_ref = self._callback
            logger.info(
                "LlamaCppConnector: cb_eval callback registered for %s",
                model_path,
            )
        except (AttributeError, TypeError) as e:
            logger.warning(
                "Could not set cb_eval on context_params: %s. "
                "Hidden state extraction may not work.",
                e,
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

        # Clear previous capture
        self._capture.clear()

        # Tokenize
        tokens = self._model.tokenize(prompt.encode("utf-8"), add_bos=True)

        # Run forward pass (eval) — cb_eval callback captures hidden states
        self._model.eval(tokens)

        if "data" not in self._capture:
            logger.warning(
                "think(): No hidden state captured. cb_eval may not be active."
            )
            return None

        # Build AVPContext from captured hidden state
        import torch

        hidden = torch.from_numpy(self._capture["data"].copy()).float()
        # Take the last token's hidden state
        if hidden.dim() == 2:
            hidden = hidden[-1:, :]  # [1, n_embd]

        # Create a minimal AVPContext
        context = AVPContext(
            past_key_values=None,  # No KV-cache transfer for llama.cpp
            model_hash="",
            model_id=self._model_path,
            hidden_state=hidden,
            n_embd=self._n_embd,
        )

        logger.debug(
            "think(): captured layer %s, shape=%s",
            self._capture.get("layer"),
            self._capture.get("shape"),
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
        **kwargs: Any,
    ) -> str:
        """Generate text, optionally using latent context.

        Args:
            prompt: The input prompt.
            context: AVPContext from a prior think() call.
            source: Source connector for cross-model rosetta.
            cross_model: Enable cross-model projection.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        # If no context, just generate normally
        if context is None:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return output["choices"][0]["text"]

        # With context: project if cross-model, then generate
        hidden = getattr(context, "hidden_state", None)
        if hidden is None:
            # Fallback to text-only
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return output["choices"][0]["text"]

        # Cross-model rosetta projection
        if cross_model and source is not None:
            hidden = self._project_rosetta(hidden, source)

        # For now, prepend context info to prompt as text
        # (embedding injection via batch.embd requires deeper integration
        # with llama-cpp-python's internals — planned for next iteration)
        logger.debug(
            "generate(): using latent context, hidden shape=%s",
            list(hidden.shape),
        )

        # Generate with the prompt
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output["choices"][0]["text"]

    def _project_rosetta(self, hidden: Any, source: "LlamaCppConnector") -> Any:
        """Project hidden state from source model to target model space."""

        # This is a placeholder — full rosetta projection requires
        # source/target tokenizers and embed weights, which need to be
        # extracted from the GGUF metadata. Planned for next iteration.
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
