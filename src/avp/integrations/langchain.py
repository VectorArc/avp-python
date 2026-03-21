"""LangChain integration for AVP latent communication.

Provides ``ChatAVP``, a LangChain ``BaseChatModel`` that uses AVP's
latent thinking under the hood. Agents exchange hidden-state context
instead of text, achieving 14-78% token savings with equal or better
accuracy.

Basic usage::

    from avp.integrations.langchain import ChatAVP

    llm = ChatAVP(model="Qwen/Qwen2.5-7B-Instruct")
    result = llm.invoke("Solve step by step: 24 * 17 + 3")

Multi-agent with latent context::

    from avp.integrations.langchain import ChatAVP
    import avp

    store = avp.ContextStore(default_ttl=300)
    researcher = ChatAVP(model="Qwen/Qwen2.5-7B-Instruct", role="think",
                         store=store, store_key="task-1")
    solver = ChatAVP(model="Qwen/Qwen2.5-7B-Instruct", role="generate",
                     store=store, store_key="task-1")

    # In a LangGraph workflow:
    researcher.invoke("Analyze this math problem: 24 * 17 + 3")
    answer = solver.invoke("Solve step by step: 24 * 17 + 3")

Cross-model::

    researcher = ChatAVP(model="Qwen/Qwen2.5-7B-Instruct", role="think",
                         store=store, store_key="task-1")
    solver = ChatAVP(model="Qwen/Qwen2.5-1.5B-Instruct", role="generate",
                     source_model="Qwen/Qwen2.5-7B-Instruct",
                     cross_model=True, store=store, store_key="task-1")

Requires: ``pip install langchain-core>=1.0``
"""

import logging
from typing import Any, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


def _messages_to_prompt(messages: List[Any]) -> str:
    """Convert LangChain messages to a single prompt string.

    Concatenates message content with role prefixes. For single-message
    inputs (the common case), returns the content directly.
    """
    if len(messages) == 1:
        return messages[0].content

    parts = []
    for msg in messages:
        role = getattr(msg, "type", "human")
        if role == "system":
            parts.append(msg.content)
        elif role == "human":
            parts.append(f"User: {msg.content}")
        elif role == "ai":
            parts.append(f"Assistant: {msg.content}")
        else:
            parts.append(msg.content)
    return "\n".join(parts)


if HAS_LANGCHAIN:

    class ChatAVP(BaseChatModel):
        """LangChain chat model that uses AVP latent thinking.

        Operates in two roles:

        - **think** (``role="think"``): Runs latent thinking steps and
          stores the context. Returns a short acknowledgment (not a full
          generation). Use this for the researcher/analyzer agent.

        - **generate** (``role="generate"``, default): Generates text,
          optionally using stored latent context from a prior think step.
          Use this for the solver/writer agent.

        When no store is provided, operates as a standard model (think +
        generate in one call via ``avp.generate()``).
        """

        model: str
        """HuggingFace model ID (e.g. 'Qwen/Qwen2.5-7B-Instruct')."""

        role: str = "generate"
        """Agent role: 'think' (store context) or 'generate' (use context)."""

        source_model: Optional[str] = None
        """Source model for cross-model rosetta (e.g. the researcher's model)."""

        cross_model: bool = False
        """Enable cross-model rosetta projection."""

        steps: int = 20
        """Number of latent thinking steps."""

        max_new_tokens: int = 512
        """Maximum tokens to generate."""

        temperature: float = 0.7
        """Sampling temperature."""

        store: Optional[Any] = None
        """AVP ContextStore for sharing latent context between agents."""

        store_key: Optional[str] = None
        """Key for storing/retrieving context in the ContextStore."""

        class Config:
            arbitrary_types_allowed = True

        @property
        def _llm_type(self) -> str:
            return "avp"

        @property
        def _identifying_params(self) -> dict:
            return {
                "model": self.model,
                "role": self.role,
                "steps": self.steps,
                "cross_model": self.cross_model,
            }

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            import avp

            prompt = _messages_to_prompt(messages)

            if self.role == "think":
                # Think-only: run latent steps, store context, return ack
                context = avp.think(
                    prompt,
                    model=self.model,
                    steps=self.steps,
                )

                # Store context if a store is provided
                if self.store is not None and self.store_key:
                    self.store.store(self.store_key, context)

                text = f"[AVP: {self.steps} latent steps completed]"
                message = AIMessage(
                    content=text,
                    additional_kwargs={"avp_role": "think", "avp_steps": self.steps},
                )

            else:
                # Generate: optionally load context, then generate text
                context = None
                if self.store is not None and self.store_key:
                    context = self.store.get(self.store_key)

                text = avp.generate(
                    prompt,
                    model=self.model,
                    source_model=self.source_model,
                    cross_model=self.cross_model,
                    steps=self.steps,
                    context=context,
                    store=self.store,
                    store_key=self.store_key,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )

                # avp.generate can return (text, metrics) tuple
                if isinstance(text, tuple):
                    text = text[0]

                message = AIMessage(
                    content=text,
                    additional_kwargs={"avp_role": "generate"},
                )

            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            # AVP doesn't support streaming — generate full response
            result = self._generate(messages, stop, run_manager, **kwargs)
            text = result.generations[0].message.content
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=text),
            )

else:

    class ChatAVP:  # type: ignore[no-redef]
        """Stub when langchain-core is not installed."""

        def __init__(self, **kwargs: Any):
            raise ImportError(
                "ChatAVP requires langchain-core. "
                "Install it with: pip install langchain-core>=1.0"
            )
