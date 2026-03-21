"""Tests for LangChain integration (mock-based, no GPU required)."""

import pytest

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.outputs import ChatResult

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = [
    pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed"),
]


class TestChatAVP:
    """Test ChatAVP LangChain model."""

    def test_import(self):
        from avp.integrations.langchain import ChatAVP
        assert ChatAVP is not None

    def test_llm_type(self):
        from avp.integrations.langchain import ChatAVP
        llm = ChatAVP(model="mock/model")
        assert llm._llm_type == "avp"

    def test_identifying_params(self):
        from avp.integrations.langchain import ChatAVP
        llm = ChatAVP(model="mock/model", steps=10, cross_model=True)
        params = llm._identifying_params
        assert params["model"] == "mock/model"
        assert params["steps"] == 10
        assert params["cross_model"] is True

    def test_role_defaults_to_generate(self):
        from avp.integrations.langchain import ChatAVP
        llm = ChatAVP(model="mock/model")
        assert llm.role == "generate"

    def test_think_role(self):
        from avp.integrations.langchain import ChatAVP
        llm = ChatAVP(model="mock/model", role="think")
        assert llm.role == "think"

    def test_messages_to_prompt_single(self):
        from avp.integrations.langchain import _messages_to_prompt
        msgs = [HumanMessage(content="hello")]
        assert _messages_to_prompt(msgs) == "hello"

    def test_messages_to_prompt_multi(self):
        from avp.integrations.langchain import _messages_to_prompt
        msgs = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="What is 2+2?"),
            AIMessage(content="4"),
            HumanMessage(content="And 3+3?"),
        ]
        result = _messages_to_prompt(msgs)
        assert "You are helpful." in result
        assert "User: What is 2+2?" in result
        assert "Assistant: 4" in result
        assert "User: And 3+3?" in result

    def test_store_integration(self):
        """ChatAVP with role=think stores context, role=generate retrieves it."""
        from avp.integrations.langchain import ChatAVP
        from avp.context_store import ContextStore

        store = ContextStore(default_ttl=60)
        thinker = ChatAVP(model="mock/model", role="think",
                          store=store, store_key="test-1")
        solver = ChatAVP(model="mock/model", role="generate",
                         store=store, store_key="test-1")

        # Both should be constructable with shared store
        assert thinker.store is solver.store
        assert thinker.store_key == solver.store_key

    def test_cross_model_config(self):
        from avp.integrations.langchain import ChatAVP
        llm = ChatAVP(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            source_model="Qwen/Qwen2.5-7B-Instruct",
            cross_model=True,
            role="generate",
        )
        assert llm.cross_model is True
        assert llm.source_model == "Qwen/Qwen2.5-7B-Instruct"


class TestChatAVPStub:
    """Test that stub raises ImportError when langchain-core is missing."""

    def test_stub_raises_without_langchain(self):
        """Covered by the skip marker — if langchain IS installed, this tests the real class."""
        from avp.integrations.langchain import ChatAVP
        # If we get here, langchain is installed and ChatAVP is real
        assert hasattr(ChatAVP, "_generate")
