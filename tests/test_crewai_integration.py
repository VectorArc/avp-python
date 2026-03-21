"""Tests for CrewAI integration (mock-based, no GPU required)."""

import pytest

try:
    from crewai.llm import BaseLLM

    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False

pytestmark = [
    pytest.mark.skipif(not HAS_CREWAI, reason="crewai not installed"),
]


class TestAVPLLM:
    """Test AVPLLM CrewAI model."""

    def test_import(self):
        from avp.integrations.crewai import AVPLLM
        assert AVPLLM is not None

    def test_is_base_llm(self):
        from avp.integrations.crewai import AVPLLM
        llm = AVPLLM(model="mock/model")
        assert isinstance(llm, BaseLLM)

    def test_provider(self):
        from avp.integrations.crewai import AVPLLM
        llm = AVPLLM(model="mock/model")
        assert llm.provider == "avp"

    def test_is_not_litellm(self):
        from avp.integrations.crewai import AVPLLM
        llm = AVPLLM(model="mock/model")
        assert llm.is_litellm is False

    def test_role_defaults_to_generate(self):
        from avp.integrations.crewai import AVPLLM
        llm = AVPLLM(model="mock/model")
        assert llm._avp_role == "generate"

    def test_think_role(self):
        from avp.integrations.crewai import AVPLLM
        llm = AVPLLM(model="mock/model", role="think")
        assert llm._avp_role == "think"

    def test_cross_model_config(self):
        from avp.integrations.crewai import AVPLLM
        llm = AVPLLM(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            source_model="Qwen/Qwen2.5-7B-Instruct",
            cross_model=True,
        )
        assert llm._avp_cross_model is True
        assert llm._avp_source_model == "Qwen/Qwen2.5-7B-Instruct"

    def test_store_integration(self):
        from avp.integrations.crewai import AVPLLM
        from avp.context_store import ContextStore

        store = ContextStore(default_ttl=60)
        thinker = AVPLLM(model="mock/model", role="think",
                         store=store, store_key="test-1")
        solver = AVPLLM(model="mock/model", role="generate",
                        store=store, store_key="test-1")

        assert thinker._avp_store is solver._avp_store
        assert thinker._avp_store_key == solver._avp_store_key

    def test_supports_stop_words(self):
        from avp.integrations.crewai import AVPLLM
        llm = AVPLLM(model="mock/model")
        assert llm.supports_stop_words() is False

    def test_context_window_size(self):
        from avp.integrations.crewai import AVPLLM
        llm = AVPLLM(model="mock/model")
        assert llm.get_context_window_size() == 32768
