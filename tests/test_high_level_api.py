"""Tests for the high-level API: think(), generate(), AVPContext."""

from unittest.mock import MagicMock

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_TRANSFORMERS),
    reason="torch and transformers required",
)


# ---- AVPContext unit tests ----


class TestAVPContext:
    def test_context_creation(self, tiny_tied_connector):
        """AVPContext holds fields correctly."""
        from avp.context import AVPContext

        # Create a small KV-cache via think
        ctx = tiny_tied_connector.think("Hello", steps=2)
        assert isinstance(ctx, AVPContext)
        assert ctx.model_hash != ""
        assert ctx.num_steps == 2
        assert ctx.seq_len > 0
        assert ctx.past_key_values is not None
        # model_family is set from identity
        assert ctx.model_family == "gpt2"

    def test_context_to_bytes_from_bytes_roundtrip(self, tiny_tied_connector):
        """Serialize and deserialize AVPContext, verify fields survive."""
        from avp.context import AVPContext

        ctx = tiny_tied_connector.think("Test roundtrip", steps=3)

        wire = ctx.to_bytes(
            session_id="test-session",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            model_id="test-model",
        )
        assert isinstance(wire, bytes)
        assert len(wire) > 0

        restored = AVPContext.from_bytes(wire, device="cpu")
        assert restored.model_hash == ctx.model_hash
        assert restored.num_steps == ctx.num_steps
        assert restored.seq_len == ctx.seq_len

    def test_context_from_bytes_preserves_extra_fields(self, tiny_tied_connector):
        """model_family, hidden_dim, num_layers survive roundtrip."""
        from avp.context import AVPContext

        ctx = tiny_tied_connector.think("Extra fields", steps=2)
        wire = ctx.to_bytes()
        restored = AVPContext.from_bytes(wire, device="cpu")

        assert restored.model_family == ctx.model_family
        assert restored.hidden_dim == ctx.hidden_dim
        assert restored.num_layers == ctx.num_layers


# ---- ABC contract tests ----


class TestABCContract:
    def test_base_connector_can_think_false(self):
        """Default can_think is False on the ABC."""
        from avp.connectors.base import EngineConnector

        # Create a minimal concrete subclass
        class MinimalConnector(EngineConnector):
            def get_model_identity(self): ...
            def extract_hidden_state(self, *a, **kw): ...
            def inject_and_generate(self, *a, **kw): ...
            def get_embedding_weights(self): ...
            def tokenize(self, text): ...
            def needs_realignment(self): ...

        c = MinimalConnector()
        assert c.can_think is False

    def test_base_connector_think_raises(self):
        """Default think() raises EngineNotAvailableError."""
        from avp.connectors.base import EngineConnector
        from avp.errors import EngineNotAvailableError

        class MinimalConnector(EngineConnector):
            def get_model_identity(self): ...
            def extract_hidden_state(self, *a, **kw): ...
            def inject_and_generate(self, *a, **kw): ...
            def get_embedding_weights(self): ...
            def tokenize(self, text): ...
            def needs_realignment(self): ...

        c = MinimalConnector()
        with pytest.raises(EngineNotAvailableError, match="think.*not supported"):
            c.think("test")

    def test_base_connector_generate_raises(self):
        """Default generate() raises NotImplementedError."""
        from avp.connectors.base import EngineConnector

        class MinimalConnector(EngineConnector):
            def get_model_identity(self): ...
            def extract_hidden_state(self, *a, **kw): ...
            def inject_and_generate(self, *a, **kw): ...
            def get_embedding_weights(self): ...
            def tokenize(self, text): ...
            def needs_realignment(self): ...

        c = MinimalConnector()
        with pytest.raises(NotImplementedError, match="generate.*not implemented"):
            c.generate("test")


# ---- HuggingFace think() tests ----


class TestHuggingFaceThink:
    def test_think_returns_context(self, tiny_tied_connector):
        """think() returns AVPContext with correct fields."""
        from avp.context import AVPContext

        ctx = tiny_tied_connector.think("What is 2+2?", steps=5)
        assert isinstance(ctx, AVPContext)
        assert ctx.model_hash == tiny_tied_connector._model_hash
        assert ctx.seq_len > 0
        assert ctx.num_steps == 5

    def test_think_can_think_true(self, tiny_tied_connector):
        """HuggingFaceConnector.can_think is True."""
        assert tiny_tied_connector.can_think is True

    def test_think_string_prompt(self, tiny_tied_connector):
        """String prompt is wrapped into user message automatically."""
        ctx = tiny_tied_connector.think("Hello world", steps=2)
        assert ctx.seq_len > 0

    def test_think_message_list_prompt(self, tiny_tied_connector):
        """List of message dicts works as prompt."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        ctx = tiny_tied_connector.think(messages, steps=2)
        assert ctx.seq_len > 0

    def test_think_chained(self, tiny_tied_connector):
        """Chaining think() calls grows seq_len and accumulates steps."""
        ctx1 = tiny_tied_connector.think("Step one", steps=3)
        ctx2 = tiny_tied_connector.think("Step two", steps=4, context=ctx1)

        assert ctx2.num_steps == 7  # 3 + 4
        assert ctx2.seq_len > ctx1.seq_len

    def test_think_incompatible_context(self, tiny_tied_connector):
        """Wrong model_hash on context raises IncompatibleModelsError."""
        from avp.context import AVPContext
        from avp.errors import IncompatibleModelsError

        fake_ctx = AVPContext(
            past_key_values=None,
            model_hash="wrong_hash",
            num_steps=1,
            seq_len=10,
        )
        with pytest.raises(IncompatibleModelsError, match="model_hash"):
            tiny_tied_connector.think("test", context=fake_ctx)

    def test_think_zero_steps_raises(self, tiny_tied_connector):
        """steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="steps must be >= 1"):
            tiny_tied_connector.think("test", steps=0)


# ---- HuggingFace generate() tests ----


class TestHuggingFaceGenerate:
    def test_generate_without_context(self, tiny_tied_connector):
        """Standalone generation returns a string."""
        result = tiny_tied_connector.generate("What is 2+2?", max_new_tokens=10)
        assert isinstance(result, str)

    def test_generate_with_context(self, tiny_tied_connector):
        """think() -> generate() pipeline returns a string."""
        ctx = tiny_tied_connector.think("Analyze this: 2+2", steps=3)
        result = tiny_tied_connector.generate("Solve it.", context=ctx, max_new_tokens=10)
        assert isinstance(result, str)

    def test_generate_incompatible_context(self, tiny_tied_connector):
        """Wrong model_hash on context raises IncompatibleModelsError."""
        from avp.context import AVPContext
        from avp.errors import IncompatibleModelsError

        fake_ctx = AVPContext(
            past_key_values=None,
            model_hash="wrong_hash",
            num_steps=1,
            seq_len=10,
        )
        with pytest.raises(IncompatibleModelsError, match="model_hash"):
            tiny_tied_connector.generate("test", context=fake_ctx)

    def test_generate_greedy(self, tiny_tied_connector):
        """do_sample=False (greedy) works."""
        result = tiny_tied_connector.generate(
            "What is 2+2?", max_new_tokens=10, do_sample=False
        )
        assert isinstance(result, str)


# ---- Full pipeline tests ----


class TestPipeline:
    def test_pipeline_tied_model(self, tiny_tied_connector):
        """Full think -> generate pipeline on GPT2 tiny (tied weights)."""
        ctx = tiny_tied_connector.think("Analyze: what is 5+3?", steps=3)
        answer = tiny_tied_connector.generate(
            "Now give the answer.", context=ctx, max_new_tokens=10
        )
        assert isinstance(answer, str)

    def test_pipeline_untied_model(self, tiny_untied_connector):
        """Full think -> generate pipeline on Llama tiny (untied weights)."""
        ctx = tiny_untied_connector.think("Analyze: what is 5+3?", steps=3)
        answer = tiny_untied_connector.generate(
            "Now give the answer.", context=ctx, max_new_tokens=10
        )
        assert isinstance(answer, str)

    def test_pipeline_with_serialization(self, tiny_tied_connector):
        """think -> to_bytes -> from_bytes -> generate round trip."""
        from avp.context import AVPContext

        ctx = tiny_tied_connector.think("Serialize test", steps=3)
        wire = ctx.to_bytes(session_id="s1", source_agent_id="a", target_agent_id="b")
        restored = AVPContext.from_bytes(wire, device="cpu")

        answer = tiny_tied_connector.generate(
            "Continue.", context=restored, max_new_tokens=10
        )
        assert isinstance(answer, str)


# ---- vLLM generate() tests (mock-based) ----


class MockHFConfig:
    def __init__(self):
        self.model_type = "qwen2"
        self._name_or_path = "Qwen/Qwen2.5-7B-Instruct"
        self.hidden_size = 3584
        self.num_hidden_layers = 28
        self.num_attention_heads = 28
        self.num_key_value_heads = 4
        self.head_dim = 128
        self.tie_word_embeddings = False
        self.vocab_size = 152064

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "_name_or_path": self._name_or_path,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "tie_word_embeddings": self.tie_word_embeddings,
            "vocab_size": self.vocab_size,
        }


class MockModelConfig:
    def __init__(self):
        self.hf_config = MockHFConfig()


class MockTokenizer:
    def __init__(self):
        self.vocab_size = 152064
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kw):
        ids = [ord(c) % (self.vocab_size - 2) + 2 for c in text]
        return {"input_ids": torch.tensor([ids])}

    def get_vocab(self):
        return {f"tok_{i}": i for i in range(1000)}


class MockLLM:
    def __init__(self):
        self.llm_engine = MagicMock()
        self.llm_engine.model_config = MockModelConfig()

    def get_tokenizer(self):
        return MockTokenizer()

    def generate(self, inputs, sampling_params=None, **kwargs):
        results = []
        for _ in inputs:
            output = MagicMock()
            output.outputs = [MagicMock(text="Mock vLLM output")]
            results.append(output)
        return results


class TestVLLMGenerate:
    @pytest.fixture
    def vllm_connector(self):
        from avp.connectors.vllm import VLLMConnector
        return VLLMConnector(engine=MockLLM())

    def test_vllm_generate_text(self, vllm_connector):
        """vLLM generate() returns a string."""
        result = vllm_connector.generate("What is 2+2?")
        assert isinstance(result, str)
        assert result == "Mock vLLM output"

    def test_vllm_generate_with_context_raises(self, vllm_connector):
        """Passing context to vLLM generate() raises EngineNotAvailableError."""
        from avp.context import AVPContext
        from avp.errors import EngineNotAvailableError

        fake_ctx = AVPContext(
            past_key_values=None,
            model_hash="any",
            num_steps=1,
            seq_len=10,
        )
        with pytest.raises(EngineNotAvailableError, match="context injection"):
            vllm_connector.generate("test", context=fake_ctx)

    def test_vllm_can_think_false(self, vllm_connector):
        """vLLM can_think is False."""
        assert vllm_connector.can_think is False

    def test_vllm_think_raises(self, vllm_connector):
        """vLLM think() raises EngineNotAvailableError."""
        from avp.errors import EngineNotAvailableError

        with pytest.raises(EngineNotAvailableError, match="think.*not supported"):
            vllm_connector.think("test")

    def test_vllm_generate_message_list(self, vllm_connector):
        """vLLM generate() accepts list of message dicts."""
        messages = [{"role": "user", "content": "Hello"}]
        result = vllm_connector.generate(messages)
        assert result == "Mock vLLM output"


# ---- from_pretrained() tests (mock-based) ----


class TestFromPretrained:
    def test_from_pretrained_auto_device_cpu(self):
        """from_pretrained auto-detects CPU when no GPU available."""
        from avp.connectors.huggingface import HuggingFaceConnector

        from transformers import GPT2Config, GPT2LMHeadModel
        from tests.conftest import MockTokenizer as ConfMockTokenizer

        # Patch AutoModel and AutoTokenizer to avoid downloads
        config = GPT2Config(vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128)
        model = GPT2LMHeadModel(config)
        tokenizer = ConfMockTokenizer(vocab_size=256)

        from unittest.mock import patch
        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("torch.cuda.is_available", return_value=False):
            connector = HuggingFaceConnector.from_pretrained("fake/model")

        assert connector.device == "cpu"
        assert connector.can_think is True


# ---- Lazy import tests ----


class TestLazyImports:
    def test_avp_context_lazy_import(self):
        """AVPContext is importable from top-level avp package."""
        import avp
        assert hasattr(avp, "AVPContext")
        from avp import AVPContext
        from avp.context import AVPContext as Direct
        assert AVPContext is Direct

    def test_huggingface_connector_lazy_import(self):
        """HuggingFaceConnector is importable from top-level avp package."""
        import avp
        from avp import HuggingFaceConnector
        from avp.connectors.huggingface import HuggingFaceConnector as Direct
        assert HuggingFaceConnector is Direct
