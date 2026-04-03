"""AutoGen integration for AVP latent communication.

Provides ``AVPChatCompletionClient``, an AutoGen ``ChatCompletionClient``
that uses AVP's latent thinking under the hood.

Usage::

    from avp.integrations.autogen import AVPChatCompletionClient
    from autogen_agentchat.agents import AssistantAgent
    import avp

    store = avp.ContextStore(default_ttl=300)

    researcher = AssistantAgent(
        "researcher",
        model_client=AVPChatCompletionClient(
            model="Qwen/Qwen2.5-7B-Instruct", role="think",
            store=store, store_key="task-1",
        ),
    )
    solver = AssistantAgent(
        "solver",
        model_client=AVPChatCompletionClient(
            model="Qwen/Qwen2.5-7B-Instruct", role="generate",
            store=store, store_key="task-1",
        ),
    )

Requires: ``pip install autogen-core``
"""

import logging
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence, Union

logger = logging.getLogger(__name__)

try:
    from autogen_core.models import (
        ChatCompletionClient,
        CreateResult,
        RequestUsage,
    )

    HAS_AUTOGEN = True
except ImportError:
    HAS_AUTOGEN = False


if HAS_AUTOGEN:

    class AVPChatCompletionClient(ChatCompletionClient):
        """AutoGen ChatCompletionClient that uses AVP latent thinking.

        Operates in two roles:

        - **think** (``role="think"``): Runs latent thinking steps and
          stores the context. Returns a short acknowledgment.

        - **generate** (``role="generate"``, default): Generates text,
          optionally using stored latent context from a prior think step.
        """

        def __init__(
            self,
            model: Optional[str] = None,
            connector: Optional[Any] = None,
            role: str = "generate",
            source_model: Optional[str] = None,
            source_connector: Optional[Any] = None,
            cross_model: bool = False,
            steps: int = 10,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            store: Optional[Any] = None,
            store_key: Optional[str] = None,
        ):
            self._avp_model = model
            self._avp_connector = connector
            self._avp_role = role
            self._avp_source_model = source_model
            self._avp_source_connector = source_connector
            self._avp_cross_model = cross_model
            self._avp_steps = steps
            self._avp_max_new_tokens = max_new_tokens
            self._avp_temperature = temperature
            self._avp_store = store
            self._avp_store_key = store_key
            self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        async def create(
            self,
            messages: Sequence[Any],
            *,
            tools: Sequence[Any] = [],
            tool_choice: Any = "auto",
            json_output: Optional[Any] = None,
            extra_create_args: Mapping[str, Any] = {},
            cancellation_token: Optional[Any] = None,
        ) -> CreateResult:
            import avp

            # Extract prompt from the last message
            prompt = self._extract_prompt(messages)

            if self._avp_role == "think":
                context = avp.think(
                    prompt,
                    model=self._avp_connector or self._avp_model,
                    steps=self._avp_steps,
                )
                if self._avp_store is not None and self._avp_store_key:
                    ctx = context.context if hasattr(context, "context") else context
                    self._avp_store.store(self._avp_store_key, ctx)
                text = f"[AVP: {self._avp_steps} latent steps completed]"
            else:
                context = None
                if self._avp_store is not None and self._avp_store_key:
                    context = self._avp_store.get(self._avp_store_key)

                text = avp.generate(
                    prompt,
                    model=self._avp_connector or self._avp_model,
                    source_model=self._avp_source_connector or self._avp_source_model,
                    cross_model=self._avp_cross_model,
                    steps=self._avp_steps,
                    context=context,
                    store=self._avp_store,
                    store_key=self._avp_store_key,
                    max_new_tokens=self._avp_max_new_tokens,
                    temperature=self._avp_temperature,
                )

            return CreateResult(
                finish_reason="stop",
                content=text,
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )

        async def create_stream(
            self,
            messages: Sequence[Any],
            *,
            tools: Sequence[Any] = [],
            tool_choice: Any = "auto",
            json_output: Optional[Any] = None,
            extra_create_args: Mapping[str, Any] = {},
            cancellation_token: Optional[Any] = None,
        ) -> AsyncGenerator[Union[str, CreateResult], None]:
            # No streaming support — yield full result
            result = await self.create(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                cancellation_token=cancellation_token,
            )
            yield result

        def count_tokens(
            self, messages: Sequence[Any], *, tools: Sequence[Any] = [],
        ) -> int:
            return sum(len(str(getattr(m, "content", m))) for m in messages) // 4

        def remaining_tokens(
            self, messages: Sequence[Any], *, tools: Sequence[Any] = [],
        ) -> int:
            return 32768 - self.count_tokens(messages, tools=tools)

        def actual_usage(self) -> RequestUsage:
            return self._total_usage

        def total_usage(self) -> RequestUsage:
            return self._total_usage

        @property
        def capabilities(self) -> Any:
            from autogen_core.models import ModelCapabilities
            return ModelCapabilities(
                vision=False,
                function_calling=False,
                json_output=False,
            )

        @property
        def model_info(self) -> Any:
            from autogen_core.models import ModelInfo, ModelFamily
            return ModelInfo(
                vision=False,
                function_calling=False,
                json_output=False,
                family=ModelFamily.UNKNOWN,
                structured_output=False,
            )

        async def close(self) -> None:
            pass

        def _extract_prompt(self, messages: Sequence[Any]) -> str:
            """Extract the last message's content as a prompt string."""
            if not messages:
                return ""
            last = messages[-1]
            if isinstance(last, str):
                return last
            content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
            return str(last)

        @classmethod
        def _from_config(cls, config: Mapping[str, Any]) -> "AVPChatCompletionClient":
            return cls(**config)

        def _to_config(self) -> Mapping[str, Any]:
            return {
                "model": self._avp_model,
                "role": self._avp_role,
                "steps": self._avp_steps,
            }

else:

    class AVPChatCompletionClient:  # type: ignore[no-redef]
        """Stub when autogen-core is not installed."""

        def __init__(self, **kwargs: Any):
            raise ImportError(
                "AVPChatCompletionClient requires autogen-core. "
                "Install it with: pip install autogen-core"
            )
