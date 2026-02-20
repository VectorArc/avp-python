"""Shared test fixtures and markers for AVP tests."""

import sys

import pytest

# Check for optional dependencies
HAS_TORCH = False
HAS_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    pass

# Custom markers
requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not installed"
)
requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)


class MockTokenizer:
    """Minimal tokenizer for integration tests (no download required).

    Implements the interface HuggingFaceConnector needs:
    pad_token_id, eos_token_id, __call__, decode, __len__.
    """

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kw):
        import torch

        ids = [ord(c) % (self.vocab_size - 2) + 2 for c in text]  # avoid 0,1 (special)
        t = torch.tensor([ids])
        return {"input_ids": t}

    def decode(self, ids, skip_special_tokens=True, **kw):
        chars = [chr(int(i) % 128) for i in ids if int(i) >= 2 or not skip_special_tokens]
        return "".join(chars).strip()

    def __len__(self):
        return self.vocab_size


@pytest.fixture
def model_config_dict():
    """A minimal model config dict for testing."""
    return {
        "model_type": "llama",
        "_name_or_path": "meta-llama/Llama-2-7b",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "tie_word_embeddings": False,
    }


@pytest.fixture
def model_config_dict_tied():
    """A model config dict with tied embeddings."""
    return {
        "model_type": "gpt2",
        "_name_or_path": "gpt2",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "head_dim": 64,
        "tie_word_embeddings": True,
    }


# --- Tiny model fixtures for integration tests ---

if HAS_TORCH and HAS_TRANSFORMERS:
    from avp.connectors import HuggingFaceConnector

    @pytest.fixture
    def tiny_tied_connector():
        """GPT2 tiny model with tied weights + mock tokenizer."""
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128,
        )
        model = GPT2LMHeadModel(config)
        tokenizer = MockTokenizer(vocab_size=256)
        return HuggingFaceConnector(model=model, tokenizer=tokenizer, device="cpu")

    @pytest.fixture
    def tiny_untied_connector():
        """Llama tiny model with untied weights + mock tokenizer."""
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config)
        tokenizer = MockTokenizer(vocab_size=256)
        return HuggingFaceConnector(model=model, tokenizer=tokenizer, device="cpu")
