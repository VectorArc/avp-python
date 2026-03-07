"""Abstract base class for AVP engine connectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from ..errors import EngineNotAvailableError
from ..types import ModelIdentity


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

    Implementations wrap specific engines (HuggingFace, vLLM, etc.)
    to provide a uniform interface for latent communication.

    High-level API (think/generate) is provided as concrete methods with
    sensible defaults. Subclasses override as needed based on engine
    capabilities.
    """

    @abstractmethod
    def get_model_identity(self) -> ModelIdentity:
        """Extract model identity for handshake."""

    @abstractmethod
    def extract_hidden_state(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any]:
        """Run forward pass and extract hidden states.

        Args:
            input_ids: Token IDs tensor [batch, seq_len].
            attention_mask: Attention mask tensor.
            past_key_values: Optional KV-cache from prior turns.

        Returns:
            Tuple of (last_hidden_state, all_hidden_states, past_key_values).
        """

    @abstractmethod
    def inject_and_generate(
        self,
        inputs_embeds: Any,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        max_new_tokens: int = 256,
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

    @abstractmethod
    def get_embedding_weights(self) -> Tuple[Any, Any]:
        """Get input and output embedding weight matrices.

        Returns:
            Tuple of (input_embedding_weight, output_embedding_weight).
        """

    @abstractmethod
    def tokenize(self, text: str) -> Any:
        """Tokenize text into input IDs.

        Args:
            text: Input text string.

        Returns:
            Token IDs tensor.
        """

    @abstractmethod
    def needs_realignment(self) -> bool:
        """Check if this model needs realignment (untied weights)."""

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
        _diagnostics: Optional[Any] = None,
    ) -> Any:
        """Generate latent context via thinking steps.

        Performs ``steps`` forward passes to build a KV-cache representing
        the model's internal reasoning about the prompt, without producing
        any text output.

        Args:
            prompt: A string (wrapped as user message) or list of chat messages.
            steps: Number of latent thinking steps.
            context: Optional AVPContext from a prior think() call to continue from.

        Returns:
            AVPContext with the accumulated KV-cache.

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
        _diagnostics: Optional[Any] = None,
    ) -> str:
        """Generate text, optionally conditioned on latent context.

        Args:
            prompt: A string (wrapped as user message) or list of chat messages.
            context: Optional AVPContext from think() to condition generation on.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to use sampling (True) or greedy decoding (False).

        Returns:
            Generated text string.

        Raises:
            NotImplementedError: If not implemented by this connector.
        """
        raise NotImplementedError(
            f"generate() is not implemented by {type(self).__name__}. "
            "Subclasses should override this method."
        )
