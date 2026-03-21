"""Tests for AutoGen integration (mock-based, no GPU required)."""

import pytest

try:
    from autogen_core.models import ChatCompletionClient, RequestUsage

    HAS_AUTOGEN = True
except ImportError:
    HAS_AUTOGEN = False

pytestmark = [
    pytest.mark.skipif(not HAS_AUTOGEN, reason="autogen-core not installed"),
]


class TestAVPChatCompletionClient:
    """Test AVPChatCompletionClient AutoGen model."""

    def test_import(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        assert AVPChatCompletionClient is not None

    def test_is_chat_completion_client(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        client = AVPChatCompletionClient(model="mock/model")
        assert isinstance(client, ChatCompletionClient)

    def test_role_defaults_to_generate(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        client = AVPChatCompletionClient(model="mock/model")
        assert client._avp_role == "generate"

    def test_think_role(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        client = AVPChatCompletionClient(model="mock/model", role="think")
        assert client._avp_role == "think"

    def test_cross_model_config(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        client = AVPChatCompletionClient(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            source_model="Qwen/Qwen2.5-7B-Instruct",
            cross_model=True,
        )
        assert client._avp_cross_model is True
        assert client._avp_source_model == "Qwen/Qwen2.5-7B-Instruct"

    def test_store_integration(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        from avp.context_store import ContextStore

        store = ContextStore(default_ttl=60)
        thinker = AVPChatCompletionClient(
            model="mock/model", role="think",
            store=store, store_key="test-1",
        )
        solver = AVPChatCompletionClient(
            model="mock/model", role="generate",
            store=store, store_key="test-1",
        )
        assert thinker._avp_store is solver._avp_store
        assert thinker._avp_store_key == solver._avp_store_key

    def test_count_tokens(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        client = AVPChatCompletionClient(model="mock/model")
        # Simple estimation: len(str) // 4
        count = client.count_tokens(["Hello world"])
        assert count > 0

    def test_usage(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        client = AVPChatCompletionClient(model="mock/model")
        usage = client.actual_usage()
        assert isinstance(usage, RequestUsage)
        assert usage.prompt_tokens == 0

    def test_extract_prompt_string(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        client = AVPChatCompletionClient(model="mock/model")
        assert client._extract_prompt(["hello"]) == "hello"

    def test_extract_prompt_empty(self):
        from avp.integrations.autogen import AVPChatCompletionClient
        client = AVPChatCompletionClient(model="mock/model")
        assert client._extract_prompt([]) == ""
