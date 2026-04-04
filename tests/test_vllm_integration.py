"""Real vLLM integration tests.

Tests VLLMConnector and AVPKVConnectorV1Dynamic with actual vLLM instances
running on GPU. Requires: Linux, CUDA GPU, vLLM >= 0.8.0, and model downloads.

Uses Qwen2.5-0.5B-Instruct (smallest Qwen, ~1GB VRAM) by default.
Set VLLM_TEST_MODEL env var to override.
"""

import os
import tempfile

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import vllm
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

pytestmark = [
    pytest.mark.requires_vllm,
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
    pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed"),
    pytest.mark.skipif(
        not HAS_TORCH or not torch.cuda.is_available(),
        reason="CUDA not available",
    ),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID = os.environ.get("VLLM_TEST_MODEL", DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Shared fixture: vLLM engine (expensive — reuse across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vllm_engine():
    """Create a vLLM LLM instance (module-scoped for reuse)."""
    engine = vllm.LLM(
        model=MODEL_ID,
        dtype="auto",
        gpu_memory_utilization=0.5,  # Leave room for 2nd model later
        enforce_eager=True,  # Faster startup, no CUDA graph compilation
        max_model_len=512,  # Keep memory low
        enable_prompt_embeds=True,  # Required for inject_and_generate
    )
    return engine


@pytest.fixture(scope="module")
def connector(vllm_engine):
    """Create VLLMConnector from the shared engine."""
    from avp.connectors.vllm import VLLMConnector
    return VLLMConnector(engine=vllm_engine)


# ---------------------------------------------------------------------------
# Test 1: Model identity extraction
# ---------------------------------------------------------------------------


def test_identity_extraction(connector):
    """VLLMConnector extracts valid model identity from vLLM engine."""
    identity = connector.get_model_identity()

    assert identity.model_family != ""
    assert identity.hidden_dim > 0
    assert identity.num_layers > 0
    assert identity.num_kv_heads > 0
    assert identity.head_dim > 0
    assert identity.model_hash != ""

    # Qwen2.5-0.5B specifics: hidden=896, layers=24, kv_heads=2
    # head_dim = hidden_size / num_attention_heads = 896 / 14 = 64
    if "0.5B" in MODEL_ID:
        assert identity.hidden_dim == 896
        assert identity.num_layers == 24
        assert identity.num_kv_heads == 2
        assert identity.head_dim == 64  # computed from hidden_size/num_attention_heads


# ---------------------------------------------------------------------------
# Test 2: Tokenization
# ---------------------------------------------------------------------------


def test_tokenization(connector):
    """VLLMConnector tokenizes text correctly."""
    tokens = connector.tokenize("Hello, world!")

    assert tokens is not None
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


# ---------------------------------------------------------------------------
# Test 3: Text generation
# ---------------------------------------------------------------------------


def test_text_generation(connector):
    """VLLMConnector generates coherent text."""
    results = connector.generate_text(
        prompts=["What is 2+2? Answer with just the number:"],
        max_tokens=10,
        temperature=0.0,
    )

    assert len(results) == 1
    assert len(results[0]) > 0
    # The answer should contain "4"
    assert "4" in results[0]


# ---------------------------------------------------------------------------
# Test 4: Batch generation
# ---------------------------------------------------------------------------


def test_batch_generation(connector):
    """VLLMConnector handles batch generation."""
    results = connector.generate_text(
        prompts=[
            "The capital of France is",
            "The capital of Japan is",
        ],
        max_tokens=10,
        temperature=0.0,
    )

    assert len(results) == 2
    assert len(results[0]) > 0
    assert len(results[1]) > 0


# ---------------------------------------------------------------------------
# Test 5: extract_hidden_state raises correctly
# ---------------------------------------------------------------------------


def test_extract_hidden_state_raises(connector):
    """extract_hidden_state raises EngineNotAvailableError for vLLM."""
    from avp.errors import EngineNotAvailableError

    with pytest.raises(EngineNotAvailableError, match="serving engine"):
        connector.extract_hidden_state(torch.zeros(1, 10))


# ---------------------------------------------------------------------------
# Test 6: Needs realignment detection
# ---------------------------------------------------------------------------


def test_needs_realignment(connector):
    """Tied/untied weight detection works."""
    result = connector.needs_realignment()
    assert isinstance(result, bool)
    # Qwen2.5-0.5B has tied weights
    if "0.5B" in MODEL_ID:
        assert result is False  # tied → no realignment needed


# ---------------------------------------------------------------------------
# Test 7: Embedding weight extraction
# ---------------------------------------------------------------------------


def test_embedding_weights(connector):
    """get_embedding_weights returns valid weight tensors."""
    input_embed, output_embed = connector.get_embedding_weights()

    assert input_embed is not None
    assert output_embed is not None
    assert input_embed.dim() == 2  # [vocab_size, hidden_dim]
    assert output_embed.dim() == 2

    if "0.5B" in MODEL_ID:
        assert input_embed.shape[1] == 896  # hidden_dim


# ---------------------------------------------------------------------------
# Test 8: AVP handshake with VLLMConnector identity
# ---------------------------------------------------------------------------


def test_handshake_with_vllm_identity(connector):
    """AVP handshake works with VLLMConnector-extracted identity."""
    from avp.handshake import CompatibilityResolver
    from avp.types import CommunicationMode

    identity = connector.get_model_identity()

    # Self-handshake should resolve to LATENT (same model_hash)
    result = CompatibilityResolver.resolve(identity, identity)
    assert result.mode == CommunicationMode.LATENT


# ---------------------------------------------------------------------------
# Test 9: Prompt embeds injection
# ---------------------------------------------------------------------------


def test_prompt_embeds_injection(connector):
    """inject_and_generate works with real vLLM prompt_embeds API."""
    # Get a real embedding for a known token
    input_embed, _ = connector.get_embedding_weights()
    _identity = connector.get_model_identity()

    # Create a small embedding sequence (3 tokens worth)
    # Use actual embedding vectors from the model
    token_ids = connector.tokenize("Hello world")
    embeds = input_embed[torch.tensor(token_ids)]  # [seq_len, hidden_dim]
    embeds = embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]

    texts, kv = connector.inject_and_generate(
        inputs_embeds=embeds,
        max_new_tokens=10,
        temperature=0.0,
    )

    assert len(texts) == 1
    assert len(texts[0]) > 0
    assert kv is None  # vLLM doesn't return KV cache


# ---------------------------------------------------------------------------
# Test 10: KV connector file-based roundtrip (no vLLM pipeline)
# ---------------------------------------------------------------------------


def test_kv_connector_file_roundtrip():
    """AVPKVConnectorV1Dynamic save/load roundtrip with real tensors."""
    from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic

    with tempfile.TemporaryDirectory() as tmpdir:

        class MockKVConfig:
            kv_connector_extra_config = {
                "avp_store_dir": tmpdir,
                "avp_latent_steps": 0,
            }

        class MockVLLMConfig:
            kv_transfer_config = MockKVConfig()

        conn = AVPKVConnectorV1Dynamic(vllm_config=MockVLLMConfig(), role=None)

        # Simulate vLLM saving KV layers during forward pass
        # Shape: [batch=1, 2(K/V), num_kv_heads=2, seq_len=32, head_dim=64]
        layer0 = torch.randn(1, 2, 2, 32, 64, dtype=torch.float16)
        layer1 = torch.randn(1, 2, 2, 32, 64, dtype=torch.float16)

        class Meta:
            request_id = "test-req-001"

        meta = Meta()
        conn.save_kv_layer("model.layers.0.self_attn", layer0, meta)
        conn.save_kv_layer("model.layers.1.self_attn", layer1, meta)
        conn.wait_for_save()

        # Store uses request_id as key since no prompt_token_ids
        assert conn._store.has_key("test-req-001")

        # Load it back
        class Ctx:
            request_id = "test-req-001"

        conn.start_load_kv(Ctx())

        # Retrieve layers
        result0 = conn.wait_for_layer_load("model.layers.0.self_attn")
        result1 = conn.wait_for_layer_load("model.layers.1.self_attn")

        assert result0 is not None
        assert result1 is not None
        assert result0.shape[0] == 2  # K and V stacked


# ---------------------------------------------------------------------------
# Test 13: Latent thinking model plugin end-to-end (requires GPU + vLLM)
# ---------------------------------------------------------------------------


def test_latent_thinking_end_to_end():
    """Model plugin adds latent steps and produces non-degenerate output."""
    from vllm.config import KVTransferConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "avp_latent_steps": 20,
                "avp_store_dir": tmpdir,
            },
        )
        engine = vllm.LLM(
            model=MODEL_ID,
            enforce_eager=True,
            max_model_len=256,
            gpu_memory_utilization=0.5,
            kv_transfer_config=ktc,
        )

        params = vllm.SamplingParams(max_tokens=50, temperature=0.0)
        outputs = engine.generate(["Solve step by step: 24 * 17 + 3"], params)

        text = outputs[0].outputs[0].text
        assert len(text) > 0
        # Should contain digits (math answer)
        assert any(c.isdigit() for c in text)


# ---------------------------------------------------------------------------
# Test 11: Full AVP codec roundtrip with vLLM-compatible tensors
# ---------------------------------------------------------------------------


def test_kv_serialize_roundtrip_vllm_shapes():
    """KV serialize/deserialize roundtrip with Qwen2.5-0.5B dimensions."""
    from avp.kv_cache import deserialize_kv_cache, serialize_kv_cache

    # Qwen2.5-0.5B dims: 24 layers, 2 KV heads, 32 tokens, 64 head_dim
    num_layers, num_kv_heads, seq_len, head_dim = 24, 2, 32, 64
    kv_cache = tuple(
        (
            torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float16),
            torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float16),
        )
        for _ in range(num_layers)
    )

    # Serialize → deserialize roundtrip
    data, header = serialize_kv_cache(kv_cache)
    assert len(data) > 0
    assert header.num_layers == num_layers
    assert header.seq_len == seq_len

    restored, header2 = deserialize_kv_cache(data)
    assert len(restored) == num_layers
    for i in range(num_layers):
        k_orig, v_orig = kv_cache[i]
        k_rest, v_rest = restored[i]
        assert torch.allclose(k_orig, k_rest)
        assert torch.allclose(v_orig, v_rest)


# ---------------------------------------------------------------------------
# Test 12: VLLMConnector model_id parameter (uses shared engine)
# ---------------------------------------------------------------------------


def test_connector_model_id_param():
    """VLLMConnector raises when neither engine nor model_id provided."""
    from avp.connectors.vllm import VLLMConnector

    with pytest.raises(ValueError, match="Provide either engine or model_id"):
        VLLMConnector()
