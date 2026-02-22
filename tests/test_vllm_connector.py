"""Tests for VLLMConnector (mock-based, no vLLM required)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---- Mock vLLM infrastructure ----

class MockHFConfig:
    """Mock HuggingFace config as exposed by vLLM."""

    def __init__(self, **kwargs):
        self.model_type = kwargs.get("model_type", "qwen2")
        self._name_or_path = kwargs.get("_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        self.hidden_size = kwargs.get("hidden_size", 3584)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 28)
        self.num_attention_heads = kwargs.get("num_attention_heads", 28)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", 128)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.vocab_size = kwargs.get("vocab_size", 152064)

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
    def __init__(self, hf_config):
        self.hf_config = hf_config


class MockTokenizer:
    """Mock tokenizer matching vLLM's tokenizer interface."""

    def __init__(self, vocab_size=152064):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kw):
        import torch
        ids = [ord(c) % (self.vocab_size - 2) + 2 for c in text]
        return {"input_ids": torch.tensor([ids])}

    def get_vocab(self):
        return {f"tok_{i}": i for i in range(min(self.vocab_size, 1000))}


class MockLLMEngine:
    def __init__(self, hf_config=None, tokenizer=None):
        self.model_config = MockModelConfig(hf_config or MockHFConfig())
        self._tokenizer = tokenizer or MockTokenizer()

    def get_tokenizer(self):
        return self._tokenizer


class MockLLM:
    """Mock vLLM LLM class."""

    def __init__(self, hf_config=None, tokenizer=None):
        self.llm_engine = MockLLMEngine(hf_config, tokenizer)

    def get_tokenizer(self):
        return self.llm_engine.get_tokenizer()

    def generate(self, inputs, sampling_params=None, **kwargs):
        """Mock generate returning MockOutput for each input."""
        results = []
        for inp in inputs:
            output = MagicMock()
            output.outputs = [MagicMock(text="Generated response")]
            results.append(output)
        return results


@pytest.fixture
def mock_engine():
    return MockLLM()


@pytest.fixture
def connector(mock_engine):
    from avp.connectors.vllm import VLLMConnector
    return VLLMConnector(engine=mock_engine)


# ---- Tests ----


def test_get_model_identity(connector):
    """get_model_identity returns correct ModelIdentity from mock config."""
    identity = connector.get_model_identity()

    assert identity.model_family == "qwen2"
    assert identity.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert identity.hidden_dim == 3584
    assert identity.num_layers == 28
    assert identity.num_kv_heads == 4
    assert identity.head_dim == 128
    assert identity.model_hash  # non-empty hash
    assert identity.tokenizer_hash  # non-empty (mock has get_vocab)


def test_get_model_identity_tied():
    """Tied-weight model identity detected correctly."""
    from avp.connectors.vllm import VLLMConnector

    config = MockHFConfig(
        model_type="gpt2",
        _name_or_path="gpt2",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        head_dim=64,
        tie_word_embeddings=True,
    )
    engine = MockLLM(hf_config=config)
    conn = VLLMConnector(engine=engine)

    identity = conn.get_model_identity()
    assert identity.model_family == "gpt2"
    assert identity.hidden_dim == 768


def test_tokenize(connector):
    """tokenize returns tensor with correct shape."""
    result = connector.tokenize("Hello world")
    assert result.dim() == 2
    assert result.shape[0] == 1
    assert result.shape[1] == len("Hello world")


def test_needs_realignment_untied(connector):
    """Untied model needs realignment."""
    assert connector.needs_realignment() is True


def test_needs_realignment_tied():
    """Tied model does not need realignment."""
    from avp.connectors.vllm import VLLMConnector

    config = MockHFConfig(tie_word_embeddings=True)
    engine = MockLLM(hf_config=config)
    conn = VLLMConnector(engine=engine)
    assert conn.needs_realignment() is False


def test_extract_hidden_state_raises(connector):
    """extract_hidden_state raises EngineNotAvailableError."""
    from avp.errors import EngineNotAvailableError

    with pytest.raises(EngineNotAvailableError, match="hidden state"):
        connector.extract_hidden_state(torch.tensor([[1, 2, 3]]))


def test_inject_and_generate_formats_prompt_embeds():
    """inject_and_generate passes prompt_embeds to engine.generate."""
    from avp.connectors.vllm import VLLMConnector

    engine = MockLLM()
    captured_inputs = []

    original_generate = engine.generate
    def tracking_generate(inputs, sampling_params=None, **kwargs):
        captured_inputs.extend(inputs)
        return original_generate(inputs, sampling_params, **kwargs)

    engine.generate = tracking_generate
    conn = VLLMConnector(engine=engine)

    # Mock vllm module for SamplingParams import
    mock_vllm = MagicMock()
    mock_vllm.SamplingParams = MagicMock(return_value=MagicMock())
    with patch.dict(sys.modules, {"vllm": mock_vllm}):
        embeds = torch.randn(2, 10, 3584)  # batch=2, seq=10, hidden=3584
        texts, past_kv = conn.inject_and_generate(embeds, max_new_tokens=50)

    assert len(captured_inputs) == 2
    assert "prompt_embeds" in captured_inputs[0]
    assert texts == ["Generated response", "Generated response"]
    assert past_kv is None


def test_generate_text():
    """generate_text passes string prompts to engine."""
    from avp.connectors.vllm import VLLMConnector

    engine = MockLLM()
    conn = VLLMConnector(engine=engine)

    mock_vllm = MagicMock()
    mock_vllm.SamplingParams = MagicMock(return_value=MagicMock())
    with patch.dict(sys.modules, {"vllm": mock_vllm}):
        texts = conn.generate_text(["What is 2+2?", "Hello"])

    assert len(texts) == 2
    assert all(t == "Generated response" for t in texts)


def test_engine_property(connector, mock_engine):
    """engine property returns underlying vLLM engine."""
    assert connector.engine is mock_engine


def test_no_engine_or_model_id_raises():
    """Must provide engine or model_id."""
    from avp.connectors.vllm import VLLMConnector

    with pytest.raises(ValueError, match="Provide either"):
        VLLMConnector()


def test_require_vllm_when_no_engine():
    """Creating with model_id when vllm not installed raises clear error."""
    from avp.connectors.vllm import VLLMConnector
    from avp.errors import EngineNotAvailableError

    # Temporarily remove vllm from modules if present
    with patch.dict(sys.modules, {"vllm": None}):
        with pytest.raises(EngineNotAvailableError, match="vllm"):
            VLLMConnector(model_id="some-model")


def test_model_hash_deterministic():
    """Same config produces same model hash."""
    from avp.connectors.vllm import VLLMConnector

    engine1 = MockLLM()
    engine2 = MockLLM()
    conn1 = VLLMConnector(engine=engine1)
    conn2 = VLLMConnector(engine=engine2)

    assert conn1.get_model_identity().model_hash == conn2.get_model_identity().model_hash


def test_different_configs_different_hash():
    """Different configs produce different model hashes."""
    from avp.connectors.vllm import VLLMConnector

    config_a = MockHFConfig(hidden_size=3584)
    config_b = MockHFConfig(hidden_size=4096)
    conn_a = VLLMConnector(engine=MockLLM(hf_config=config_a))
    conn_b = VLLMConnector(engine=MockLLM(hf_config=config_b))

    assert conn_a.get_model_identity().model_hash != conn_b.get_model_identity().model_hash


def test_hf_config_extraction_fallback():
    """Config extraction works with direct model_config access."""
    from avp.connectors.vllm import VLLMConnector

    # Engine without llm_engine but with direct model_config
    engine = MagicMock(spec=[])
    engine.llm_engine = MagicMock(spec=[])
    engine.llm_engine.model_config = MockModelConfig(MockHFConfig())
    engine.llm_engine.get_tokenizer = lambda: MockTokenizer()
    engine.get_tokenizer = lambda: MockTokenizer()

    conn = VLLMConnector(engine=engine)
    assert conn.get_model_identity().model_family == "qwen2"
