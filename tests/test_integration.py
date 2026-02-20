"""End-to-end integration tests for the AVP latent communication pipeline.

Tests the full pipeline: handshake → extract → serialize → encode → transport →
decode → deserialize → inject → generate. Uses tiny random-weight models
(no downloads required).
"""

import threading
import time

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

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
    pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed"),
]


# ---------------------------------------------------------------------------
# Test 1: KV-cache survives serialize → AVP encode → decode → deserialize
# ---------------------------------------------------------------------------


def test_kv_cache_roundtrip_through_avp_codec():
    """KV-cache survives serialize → AVP encode → AVP decode → deserialize."""
    from avp.codec import decode, encode_kv_cache
    from avp.kv_cache import deserialize_kv_cache, serialize_kv_cache
    from avp.types import AVPMetadata, DataType, PayloadType

    num_layers, num_kv_heads, seq_len, head_dim = 2, 4, 8, 16

    # Build a fake KV-cache (legacy tuple format)
    kv_cache = tuple(
        (
            torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float32),
            torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float32),
        )
        for _ in range(num_layers)
    )

    # Serialize
    kv_bytes, kv_header = serialize_kv_cache(kv_cache)
    assert kv_header.num_layers == num_layers
    assert kv_header.seq_len == seq_len

    # Build AVP metadata and encode
    metadata = AVPMetadata(
        model_id="test-model",
        hidden_dim=num_kv_heads * head_dim,
        num_layers=num_layers,
        payload_type=PayloadType.KV_CACHE,
        dtype=DataType.FLOAT32,
    )
    avp_binary = encode_kv_cache(kv_bytes, metadata)

    # Decode AVP message
    msg = decode(avp_binary)
    assert msg.metadata.payload_type == PayloadType.KV_CACHE
    assert msg.metadata.hidden_dim == num_kv_heads * head_dim
    assert msg.metadata.num_layers == num_layers
    assert msg.header.is_kv_cache is True

    # Deserialize KV-cache from payload
    restored_kv, restored_header = deserialize_kv_cache(msg.payload)
    assert restored_header.num_layers == num_layers
    assert restored_header.seq_len == seq_len
    assert len(restored_kv) == num_layers

    for layer_idx in range(num_layers):
        orig_k, orig_v = kv_cache[layer_idx]
        rest_k, rest_v = restored_kv[layer_idx]
        assert orig_k.shape == rest_k.shape
        assert orig_v.shape == rest_v.shape
        torch.testing.assert_close(orig_k, rest_k, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(orig_v, rest_v, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test 2: Full pipeline in-process — tied model (GPT2)
# ---------------------------------------------------------------------------


def test_full_pipeline_in_process_tied(tiny_tied_connector):
    """Full pipeline with tied model (GPT2), in-process (no HTTP).

    Verifies: extract → serialize → encode → decode → deserialize → forward pass.
    KV-cache data survives the full roundtrip and the model can continue from it.
    """
    from avp.codec import decode, encode_kv_cache
    from avp.kv_cache import deserialize_kv_cache, legacy_to_dynamic_cache, serialize_kv_cache
    from avp.realign import compute_target_norm, normalize_to_target
    from avp.types import AVPMetadata, DataType, PayloadType

    connector = tiny_tied_connector

    # Verify tied weights → no realignment needed
    assert connector.needs_realignment() is False

    # Agent A: tokenize and extract
    input_ids = connector.tokenize("hello world test")
    last_hidden, all_hidden, past_kv = connector.extract_hidden_state(input_ids)

    assert last_hidden.shape[0] == 1  # batch=1
    assert last_hidden.shape[1] == 64  # hidden_dim

    # Agent A: serialize KV-cache
    kv_bytes, kv_header = serialize_kv_cache(past_kv)

    # Agent A: encode into AVP binary
    metadata = AVPMetadata(
        model_id="tiny-gpt2",
        hidden_dim=64,
        num_layers=kv_header.num_layers,
        payload_type=PayloadType.KV_CACHE,
        dtype=DataType.FLOAT32,
    )
    avp_binary = encode_kv_cache(kv_bytes, metadata)

    # Agent B: decode AVP message
    msg = decode(avp_binary)
    assert msg.metadata.payload_type == PayloadType.KV_CACHE

    # Agent B: deserialize KV-cache
    restored_kv, restored_header = deserialize_kv_cache(msg.payload)
    assert restored_header.num_layers == kv_header.num_layers
    assert restored_header.seq_len == kv_header.seq_len

    # Verify KV-cache data integrity (values match after roundtrip)
    from avp.kv_cache import dynamic_cache_to_legacy

    original_kv = dynamic_cache_to_legacy(past_kv)
    for layer_idx in range(kv_header.num_layers):
        orig_k, orig_v = original_kv[layer_idx]
        rest_k, rest_v = restored_kv[layer_idx]
        torch.testing.assert_close(orig_k, rest_k, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(orig_v, rest_v, atol=1e-3, rtol=1e-3)

    # Agent B: do a forward pass with the deserialized KV-cache to prove it's usable
    dynamic_cache = legacy_to_dynamic_cache(restored_kv)

    target_norm = compute_target_norm(connector.model, device="cpu")
    normalized_hidden = normalize_to_target(last_hidden, target_norm)
    inputs_embeds = normalized_hidden.unsqueeze(1)  # [B, 1, D]

    # Forward pass with inputs_embeds + restored KV-cache
    total_len = inputs_embeds.shape[1] + kv_header.seq_len
    attention_mask = torch.ones((1, total_len), dtype=torch.long)

    with torch.no_grad():
        outputs = connector.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=dynamic_cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    # Model produced logits and updated KV-cache
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 1  # one new position
    assert outputs.logits.shape[2] == 256  # vocab_size
    assert outputs.hidden_states is not None


# ---------------------------------------------------------------------------
# Test 3: Full pipeline in-process — untied model (Llama)
# ---------------------------------------------------------------------------


def test_full_pipeline_in_process_untied(tiny_untied_connector):
    """Full pipeline with untied model (Llama), exercises realignment.

    Verifies: extract → realign → serialize → encode → decode → deserialize →
    forward pass with restored KV-cache + realigned hidden state.
    """
    from avp.codec import decode, encode_kv_cache
    from avp.kv_cache import (
        deserialize_kv_cache,
        dynamic_cache_to_legacy,
        legacy_to_dynamic_cache,
        serialize_kv_cache,
    )
    from avp.realign import apply_realignment
    from avp.types import AVPMetadata, DataType, PayloadType

    connector = tiny_untied_connector

    # Verify untied weights → realignment needed
    assert connector.needs_realignment() is True

    # Agent A: tokenize and extract
    input_ids = connector.tokenize("hello world test")
    last_hidden, all_hidden, past_kv = connector.extract_hidden_state(input_ids)

    assert last_hidden.shape[0] == 1
    assert last_hidden.shape[1] == 64

    # Compute realignment
    connector._ensure_realignment()
    assert connector._w_realign is not None
    assert connector._target_norm is not None

    # Apply realignment to hidden state
    aligned_hidden = apply_realignment(
        last_hidden, connector._w_realign, connector._target_norm
    )

    # Agent A: serialize KV-cache
    kv_bytes, kv_header = serialize_kv_cache(past_kv)

    # Agent A: encode into AVP binary
    metadata = AVPMetadata(
        model_id="tiny-llama",
        hidden_dim=64,
        num_layers=kv_header.num_layers,
        payload_type=PayloadType.KV_CACHE,
        dtype=DataType.FLOAT32,
    )
    avp_binary = encode_kv_cache(kv_bytes, metadata)

    # Agent B: decode → deserialize
    msg = decode(avp_binary)
    restored_kv, restored_header = deserialize_kv_cache(msg.payload)
    assert restored_header.num_layers == kv_header.num_layers

    # Verify KV-cache data integrity
    original_kv = dynamic_cache_to_legacy(past_kv)
    for layer_idx in range(kv_header.num_layers):
        orig_k, orig_v = original_kv[layer_idx]
        rest_k, rest_v = restored_kv[layer_idx]
        torch.testing.assert_close(orig_k, rest_k, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(orig_v, rest_v, atol=1e-3, rtol=1e-3)

    # Agent B: forward pass with restored KV-cache + realigned hidden state
    dynamic_cache = legacy_to_dynamic_cache(restored_kv)
    inputs_embeds = aligned_hidden.unsqueeze(1)  # [B, 1, D]

    total_len = inputs_embeds.shape[1] + kv_header.seq_len
    attention_mask = torch.ones((1, total_len), dtype=torch.long)

    with torch.no_grad():
        outputs = connector.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=dynamic_cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 1
    assert outputs.logits.shape[2] == 256
    assert outputs.hidden_states is not None


# ---------------------------------------------------------------------------
# Test 4: LatentMAS-style latent steps pipeline
# ---------------------------------------------------------------------------


def test_latent_steps_pipeline(tiny_tied_connector):
    """LatentMAS-style: Agent A does latent steps, transfers KV to Agent B.

    Verifies: latent_steps → serialize → encode → decode → deserialize →
    forward pass with accumulated KV-cache.
    """
    from avp.codec import decode, encode_kv_cache
    from avp.kv_cache import deserialize_kv_cache, legacy_to_dynamic_cache, serialize_kv_cache
    from avp.realign import compute_target_norm, normalize_to_target
    from avp.types import AVPMetadata, DataType, PayloadType

    connector = tiny_tied_connector
    latent_steps = 3

    # Agent A: tokenize
    input_ids = connector.tokenize("think about this")
    initial_seq_len = input_ids.shape[1]

    # Agent A: run latent steps (accumulates KV-cache)
    accumulated_kv = connector.generate_latent_steps(
        input_ids=input_ids, latent_steps=latent_steps
    )

    # Verify KV-cache grew: initial prompt tokens + latent_steps
    from avp.connectors.huggingface import _past_length

    kv_seq_len = _past_length(accumulated_kv)
    assert kv_seq_len == initial_seq_len + latent_steps

    # Agent A: serialize → encode
    kv_bytes, kv_header = serialize_kv_cache(accumulated_kv)
    assert kv_header.seq_len == initial_seq_len + latent_steps

    metadata = AVPMetadata(
        model_id="tiny-gpt2",
        hidden_dim=64,
        num_layers=kv_header.num_layers,
        payload_type=PayloadType.KV_CACHE,
        dtype=DataType.FLOAT32,
    )
    avp_binary = encode_kv_cache(kv_bytes, metadata)

    # Agent B: decode → deserialize
    msg = decode(avp_binary)
    restored_kv, restored_header = deserialize_kv_cache(msg.payload)
    assert restored_header.seq_len == initial_seq_len + latent_steps

    # Agent B: forward pass with the accumulated KV-cache
    dynamic_cache = legacy_to_dynamic_cache(restored_kv)

    # Get a hidden state to inject
    last_hidden, _, _ = connector.extract_hidden_state(input_ids)
    target_norm = compute_target_norm(connector.model, device="cpu")
    normalized = normalize_to_target(last_hidden, target_norm)
    inputs_embeds = normalized.unsqueeze(1)  # [B, 1, D]

    total_len = inputs_embeds.shape[1] + kv_header.seq_len
    attention_mask = torch.ones((1, total_len), dtype=torch.long)

    with torch.no_grad():
        outputs = connector.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=dynamic_cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    # Model accepted the accumulated KV-cache and produced output
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 1
    assert outputs.logits.shape[2] == 256
    # KV-cache grew by 1 (the injected embed)
    new_kv_len = _past_length(outputs.past_key_values)
    assert new_kv_len == kv_header.seq_len + 1


# ---------------------------------------------------------------------------
# Test 5: Handshake — same model → LATENT mode
# ---------------------------------------------------------------------------


def test_handshake_same_model_resolves_latent(tiny_tied_connector):
    """Two connectors wrapping the same model → handshake → LATENT."""
    from avp.handshake import CompatibilityResolver
    from avp.types import CommunicationMode

    id_a = tiny_tied_connector.get_model_identity()
    id_b = tiny_tied_connector.get_model_identity()

    session_info = CompatibilityResolver.resolve(id_a, id_b)
    assert session_info.mode == CommunicationMode.LATENT
    assert session_info.session_id  # non-empty


# ---------------------------------------------------------------------------
# Test 6: Handshake — different models → JSON mode
# ---------------------------------------------------------------------------


def test_handshake_different_models_resolves_json(
    tiny_tied_connector, tiny_untied_connector
):
    """Tied (GPT2) vs untied (Llama) → handshake → JSON."""
    from avp.handshake import CompatibilityResolver
    from avp.types import CommunicationMode

    id_a = tiny_tied_connector.get_model_identity()
    id_b = tiny_untied_connector.get_model_identity()

    # Different model families should resolve to JSON
    assert id_a.model_family != id_b.model_family

    session_info = CompatibilityResolver.resolve(id_a, id_b)
    assert session_info.mode == CommunicationMode.JSON


# ---------------------------------------------------------------------------
# Test 7: Full pipeline over HTTP/2
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def integration_server():
    """AVP server for integration tests (port 9124)."""
    import uvicorn

    from avp.codec import decode as avp_decode
    from avp.handshake import HelloMessage, extract_model_identity
    from avp.kv_cache import deserialize_kv_cache
    from avp.transport import create_app
    from avp.types import PayloadType

    # Server's model identity (tiny GPT2-like, matches tiny_tied_connector)
    from transformers import GPT2Config

    config = GPT2Config(vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128)
    server_identity = extract_model_identity(config)

    async def handshake_handler(hello: HelloMessage) -> dict:
        return {
            "agent_id": "integration-server",
            "identity": server_identity.to_dict(),
            "session_id": "integration-test-session",
        }

    async def transmit_handler(msg) -> dict:
        info = {
            "payload_type": msg.metadata.payload_type.value,
            "hidden_dim": msg.metadata.hidden_dim,
            "num_layers": msg.metadata.num_layers,
        }
        if msg.metadata.payload_type == PayloadType.KV_CACHE:
            restored_kv, header = deserialize_kv_cache(msg.payload)
            info.update({
                "kv_num_layers": header.num_layers,
                "kv_seq_len": header.seq_len,
                "kv_num_kv_heads": header.num_kv_heads,
                "kv_head_dim": header.head_dim,
            })
        return info

    async def text_handler(msg) -> dict:
        return {
            "content": msg.content,
            "source_agent_id": msg.source_agent_id,
        }

    app = create_app(
        transmit_handler,
        text_handler=text_handler,
        handshake_handler=handshake_handler,
    )

    config = uvicorn.Config(app, host="127.0.0.1", port=9124, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import httpx

    for _ in range(50):
        try:
            httpx.get("http://127.0.0.1:9124/health", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("Integration server did not start in time")

    yield "http://127.0.0.1:9124", server_identity

    server.should_exit = True
    thread.join(timeout=5)


def test_full_pipeline_over_http(integration_server, tiny_tied_connector):
    """Full pipeline: KV-cache serialized → AVP encode → HTTP → server decodes."""
    from avp.kv_cache import serialize_kv_cache
    from avp.transport import AVPClient
    from avp.types import AVPMetadata, CommunicationMode, DataType, PayloadType

    url, server_identity = integration_server
    connector = tiny_tied_connector

    # Client handshakes with server (same model → LATENT)
    client_identity = connector.get_model_identity()

    with AVPClient(url, agent_id="integration-client", model_identity=client_identity) as client:
        session_info = client.handshake()
        assert session_info.mode == CommunicationMode.LATENT

        # Extract and serialize KV-cache
        input_ids = connector.tokenize("http transport test")
        _, _, past_kv = connector.extract_hidden_state(input_ids)
        kv_bytes, kv_header = serialize_kv_cache(past_kv)

        # Build metadata and transmit
        metadata = AVPMetadata(
            model_id="tiny-gpt2",
            hidden_dim=64,
            num_layers=kv_header.num_layers,
            payload_type=PayloadType.KV_CACHE,
            dtype=DataType.FLOAT32,
            source_agent_id="integration-client",
        )
        resp = client.transmit(kv_bytes, metadata)

    data = resp.json()
    assert data["success"] is True
    assert data["payload_type"] == PayloadType.KV_CACHE.value
    assert data["kv_num_layers"] == kv_header.num_layers
    assert data["kv_seq_len"] == kv_header.seq_len
    assert data["kv_num_kv_heads"] == kv_header.num_kv_heads
    assert data["kv_head_dim"] == kv_header.head_dim


# ---------------------------------------------------------------------------
# Test 8: JSON fallback over HTTP
# ---------------------------------------------------------------------------


def test_json_fallback_over_http(integration_server, tiny_untied_connector):
    """When handshake resolves to JSON, agents communicate via text."""
    from avp.fallback import JSONMessage
    from avp.transport import AVPClient
    from avp.types import CommunicationMode

    url, server_identity = integration_server

    # Client uses a different model (Llama) → should get JSON mode
    client_identity = tiny_untied_connector.get_model_identity()
    assert client_identity.model_family != server_identity.model_family

    with AVPClient(url, agent_id="json-client", model_identity=client_identity) as client:
        session_info = client.handshake()
        assert session_info.mode == CommunicationMode.JSON

        # Send text via JSON fallback
        msg = JSONMessage(
            session_id=session_info.session_id,
            source_agent_id="json-client",
            target_agent_id="integration-server",
            content="Hello from JSON fallback!",
        )
        resp = client.send_text(msg)

    data = resp.json()
    assert data["success"] is True
    assert data["content"] == "Hello from JSON fallback!"
    assert data["source_agent_id"] == "json-client"
