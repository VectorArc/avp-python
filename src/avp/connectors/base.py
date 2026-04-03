"""Abstract base class for AVP engine connectors.

Extension policy
----------------
New methods added to ``EngineConnector`` will **always** have default
implementations.  Existing abstract methods will not be removed or have
their signatures changed.  This guarantees that any connector written
against v1.0 will work on v1.x without modification.

Minimal connector
-----------------
A minimal connector need only implement ``get_model_identity()`` and
override ``generate()``.  All other methods have concrete defaults that
raise ``NotImplementedError`` with descriptive messages.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..errors import EngineNotAvailableError
from ..types import ModelIdentity, PayloadType

# Type aliases for engine-agnostic tensors.
# These are Any at runtime (torch is optional) but document intent.
TensorLike = Any
"""A tensor-like object: torch.Tensor, numpy.ndarray, or similar."""

KVCache = Any
"""KV-cache: DynamicCache, tuple of (K, V) tuples, or None."""


def _render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Render chat messages using the tokenizer's chat template, with fallback."""
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    segments = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
    segments.append("<|assistant|>")
    return "\n".join(segments)


def _tokenize_prompt(
    tokenizer: Any, prompt_text: str, device: str
) -> Tuple[Any, Any]:
    """Tokenize rendered prompt, returning (input_ids, attention_mask)."""
    encoded = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        import torch

        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


class EngineConnector(ABC):
    """Abstract interface for model inference backends.

    Implementations wrap specific engines (HuggingFace, vLLM, llama.cpp,
    Ollama) to provide a uniform interface for latent communication.

    **Extension policy:** New methods added to this class will always
    have default implementations.  Subclasses need only implement
    :meth:`get_model_identity` and override :meth:`generate`.  All other
    methods have sensible defaults.

    **Capability model:** Check :attr:`can_think` before calling
    :meth:`think`.  Cross-model parameters (``source=``, ``cross_model=``)
    are connector-specific and not part of this ABC.

    **Factory methods** are connector-specific (``from_pretrained()``,
    ``from_ollama()``, etc.) and not part of this interface.
    """

    @abstractmethod
    def get_model_identity(self) -> ModelIdentity:
        """Extract model identity for handshake."""

    # --- Model introspection (read-only properties) ---

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension (embedding size) of the model.

        Delegates to :meth:`get_model_identity` by default.
        Returns 0 if the connector cannot determine the value.
        """
        return self.get_model_identity().hidden_dim

    @property
    def num_layers(self) -> int:
        """Number of transformer layers.

        Delegates to :meth:`get_model_identity` by default.
        Returns 0 if the connector cannot determine the value.
        """
        return self.get_model_identity().num_layers

    @property
    def context_length(self) -> Optional[int]:
        """Maximum context window size in tokens.

        Returns ``None`` if the connector cannot determine the value
        (e.g., remote engines without config access).
        """
        return None

    @property
    def vocab_size(self) -> Optional[int]:
        """Vocabulary size of the model's tokenizer.

        Returns ``None`` if unknown.
        """
        return None

    @property
    def device(self) -> str:
        """Device where model inference runs.

        Returns a string like ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or
        ``"mps"``.  Returns ``"cpu"`` for engines where device placement
        is managed internally (e.g., llama.cpp GPU offloading).

        Returns ``"remote"`` for remote serving engines (e.g., vLLM server).

        Note: Returns :class:`str`, not ``torch.device``, because torch
        is an optional dependency.
        """
        return "cpu"

    @property
    def dtype(self) -> str:
        """Primary computation dtype as a string.

        Returns a string like ``"float32"``, ``"float16"``, ``"bfloat16"``,
        or ``"auto"`` if the engine manages dtype internally.

        Note: Returns :class:`str`, not ``torch.dtype``, because torch
        is an optional dependency.
        """
        return "float32"

    @property
    def has_tokenizer(self) -> bool:
        """Whether this connector exposes tokenization capabilities.

        When ``True``, :meth:`tokenize`, :meth:`detokenize`, and
        :meth:`apply_chat_template` are available.  Check this before
        calling tokenization methods to avoid ``NotImplementedError``.
        """
        return False

    # --- Tokenization ---

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into a list of token IDs.

        Does **not** add special tokens (BOS/EOS).  Use
        :meth:`apply_chat_template` to produce a properly framed prompt
        before tokenizing.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.

        Raises:
            NotImplementedError: If this connector doesn't expose tokenization.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose tokenization."
        )

    def detokenize(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a text string.

        Args:
            token_ids: List of token IDs to decode.

        Returns:
            Decoded text string.

        Raises:
            NotImplementedError: If this connector doesn't expose detokenization.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose detokenization."
        )

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render chat messages using the model's chat template.

        Applies the model's native chat template (ChatML, Llama, Mistral,
        etc.) to produce a formatted prompt string ready for tokenization.

        Args:
            messages: List of ``{"role": "...", "content": "..."}`` dicts.
            add_generation_prompt: Whether to append the assistant turn
                start marker (default: ``True``).

        Returns:
            Formatted prompt string.

        Raises:
            NotImplementedError: If this connector doesn't support chat templates.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support chat templates."
        )

    @property
    def stop_token_ids(self) -> Set[int]:
        """Token IDs that signal end-of-generation.

        Includes EOS, end-of-turn markers, and model-specific stop tokens
        derived from the chat template.

        Implementations SHOULD cache this value after first computation.
        Callers MAY access this property repeatedly in hot paths.

        Returns:
            Set of stop token IDs.  Empty set if unknown.
        """
        return set()

    @property
    def stop_strings(self) -> List[str]:
        """Text strings that signal end-of-generation.

        Includes end-of-turn markers and multi-turn hallucination guards.
        Used for text-based stop detection in custom generation loops.

        Implementations SHOULD cache this value after first computation.
        Callers MAY access this property repeatedly in hot paths.

        Returns:
            List of stop strings.  Empty list if unknown.
        """
        return []

    # --- Low-level API (optional — override for engines with tensor access) ---

    def extract_hidden_state(
        self,
        input_ids: TensorLike,
        attention_mask: Optional[TensorLike] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> Tuple[TensorLike, TensorLike, KVCache]:
        """Run forward pass and extract hidden states.

        Args:
            input_ids: Token IDs tensor [batch, seq_len].
            attention_mask: Attention mask tensor.
            past_key_values: Optional KV-cache from prior turns.

        Returns:
            Tuple of (last_hidden_state, all_hidden_states, past_key_values).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose raw hidden states. "
            "Use think() for latent context extraction."
        )

    def inject_and_generate(
        self,
        inputs_embeds: TensorLike,
        attention_mask: Optional[TensorLike] = None,
        past_key_values: Optional[KVCache] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[Any, Any]:
        """Generate text from injected embeddings.

        Args:
            inputs_embeds: Embedding tensor [batch, seq_len, hidden_dim].
            attention_mask: Attention mask tensor.
            past_key_values: Optional KV-cache.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Tuple of (generated_text_list, past_key_values).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support embedding injection. "
            "Use generate(prompt, context=) instead."
        )

    def get_embedding_weights(self) -> Tuple[Optional[TensorLike], Optional[TensorLike]]:
        """Get input and output embedding weight matrices.

        Returns:
            Tuple of (input_embedding_weight, output_embedding_weight).
            Returns (None, None) if not available.
        """
        return (None, None)

    def needs_realignment(self) -> bool:
        """Check if this model needs realignment (untied weights)."""
        return False

    # --- High-level API ---

    @property
    def can_think(self) -> bool:
        """Whether this connector supports latent thinking steps (think()).

        Only connectors with full hidden state access (e.g. HuggingFaceConnector)
        return True. Check this before calling think().
        """
        return False

    def think(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        steps: int = 20,
        context: Optional[Any] = None,
        output: PayloadType = PayloadType.AUTO,
        **kwargs: Any,
    ) -> Any:
        """Generate latent context via thinking steps.

        Performs ``steps`` forward passes to build a KV-cache representing
        the model's internal reasoning about the prompt, without producing
        any text output.

        Args:
            prompt: A string (wrapped as user message) or list of chat messages.
            steps: Number of latent thinking steps.
            context: Optional AVPContext from a prior think() call to continue from.
            output: What to include in the returned context:

                ``PayloadType.AUTO`` (default): let the system decide.
                    Currently resolves to ``KV_CACHE`` for same-model
                    and ``HIDDEN_STATE`` for cross-model.

                ``PayloadType.KV_CACHE``: full KV-cache + hidden state.
                    Best for same-model, same-process transfer.

                ``PayloadType.HIDDEN_STATE``: only the last hidden state ``[1, D]``.
                    KV-cache is freed immediately, reducing VRAM.
                    Use for cross-process, bandwidth-constrained, or
                    structured tasks where a single vector suffices.
            **kwargs: Connector-specific options.

        Returns:
            AVPContext with the accumulated KV-cache or hidden state only.

        Raises:
            EngineNotAvailableError: If this connector doesn't support latent thinking.
        """
        raise EngineNotAvailableError(
            f"{type(self).__name__} (think() is not supported. "
            "Latent thinking requires hidden state access. "
            "Use HuggingFaceConnector for latent communication, "
            "or check connector.can_think before calling.)"
        )

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        context: Optional[Any] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate text, optionally conditioned on latent context.

        Args:
            prompt: A string (wrapped as user message) or list of chat messages.
            context: Optional AVPContext from think() to condition generation on.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to use sampling (True) or greedy decoding (False).
            **kwargs: Connector-specific options (e.g. ``source=``,
                ``cross_model=`` for rosetta-capable connectors).

        Returns:
            Generated text string.

        Raises:
            NotImplementedError: If not implemented by this connector.
        """
        raise NotImplementedError(
            f"generate() is not implemented by {type(self).__name__}. "
            "Subclasses should override this method."
        )
