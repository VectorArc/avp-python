"""Abstract base class for AVP engine connectors."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from ..types import ModelIdentity


class EngineConnector(ABC):
    """Abstract interface for model inference backends.

    Implementations wrap specific engines (HuggingFace, vLLM, etc.)
    to provide a uniform interface for latent communication.
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
