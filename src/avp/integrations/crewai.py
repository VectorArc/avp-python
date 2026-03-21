"""CrewAI integration for AVP latent communication.

Provides ``AVPLLM``, a CrewAI ``BaseLLM`` that uses AVP's latent
thinking under the hood.

Usage::

    from avp.integrations.crewai import AVPLLM
    from crewai import Agent, Task, Crew
    import avp

    store = avp.ContextStore(default_ttl=300)

    researcher = Agent(
        role="Researcher",
        goal="Analyze math problems",
        llm=AVPLLM(model="Qwen/Qwen2.5-7B-Instruct", role="think",
                    store=store, store_key="task-1"),
    )
    solver = Agent(
        role="Solver",
        goal="Solve math problems step by step",
        llm=AVPLLM(model="Qwen/Qwen2.5-7B-Instruct", role="generate",
                    store=store, store_key="task-1"),
    )

Requires: ``pip install crewai``
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from crewai.llm import BaseLLM

    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False


if HAS_CREWAI:

    class AVPLLM(BaseLLM):
        """CrewAI LLM that uses AVP latent thinking.

        Operates in two roles:

        - **think** (``role="think"``): Runs latent thinking steps and
          stores the context. Returns a short acknowledgment.

        - **generate** (``role="generate"``, default): Generates text,
          optionally using stored latent context from a prior think step.
        """

        def __init__(
            self,
            model: str,
            role: str = "generate",
            source_model: Optional[str] = None,
            cross_model: bool = False,
            steps: int = 20,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            store: Optional[Any] = None,
            store_key: Optional[str] = None,
        ):
            self._avp_model = model
            self._avp_role = role
            self._avp_source_model = source_model
            self._avp_cross_model = cross_model
            self._avp_steps = steps
            self._avp_max_new_tokens = max_new_tokens
            self._avp_temperature = temperature
            self._avp_store = store
            self._avp_store_key = store_key

        @property
        def provider(self) -> str:
            return "avp"

        @property
        def is_litellm(self) -> bool:
            return False

        def call(
            self,
            messages: Any,
            tools: Any = None,
            callbacks: Any = None,
            available_functions: Any = None,
            from_task: Any = None,
            from_agent: Any = None,
            response_model: Any = None,
        ) -> str:
            import avp

            # Extract prompt from messages
            if isinstance(messages, str):
                prompt = messages
            elif isinstance(messages, list) and messages:
                # CrewAI messages are dicts or LLMMessage objects
                last = messages[-1]
                if isinstance(last, dict):
                    prompt = last.get("content", str(last))
                elif hasattr(last, "content"):
                    prompt = last.content
                else:
                    prompt = str(last)
            else:
                prompt = str(messages)

            if self._avp_role == "think":
                context = avp.think(
                    prompt,
                    model=self._avp_model,
                    steps=self._avp_steps,
                )
                if self._avp_store is not None and self._avp_store_key:
                    self._avp_store.store(self._avp_store_key, context)
                return f"[AVP: {self._avp_steps} latent steps completed]"

            else:
                context = None
                if self._avp_store is not None and self._avp_store_key:
                    context = self._avp_store.get(self._avp_store_key)

                result = avp.generate(
                    prompt,
                    model=self._avp_model,
                    source_model=self._avp_source_model,
                    cross_model=self._avp_cross_model,
                    steps=self._avp_steps,
                    context=context,
                    store=self._avp_store,
                    store_key=self._avp_store_key,
                    max_new_tokens=self._avp_max_new_tokens,
                    temperature=self._avp_temperature,
                )
                if isinstance(result, tuple):
                    result = result[0]
                return result

        def supports_stop_words(self) -> bool:
            return False

        def supports_multimodal(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 32768

        def get_token_usage_summary(self) -> str:
            return ""

else:

    class AVPLLM:  # type: ignore[no-redef]
        """Stub when crewai is not installed."""

        def __init__(self, **kwargs: Any):
            raise ImportError(
                "AVPLLM requires crewai. "
                "Install it with: pip install crewai"
            )
