"""Tests for AVP transport layer (client + server)."""

import threading
import time

import numpy as np
import pytest
import uvicorn

import avp
from avp.fallback import JSONMessage
from avp.transport import AVPClient, create_app
from avp.types import AVPMetadata, DataType, PayloadType
from avp.utils import embedding_to_bytes


# --- Server handler for tests ---


async def echo_handler(msg: avp.AVPMessage) -> dict:
    """Echo back metadata about the received message."""
    return {
        "model_id": msg.metadata.model_id,
        "embedding_dim": msg.metadata.embedding_dim,
        "data_type": msg.metadata.data_type,
        "agent_id": msg.metadata.agent_id or "",
        "compressed": msg.header.compressed,
    }


async def text_handler(msg: JSONMessage) -> dict:
    """Echo back JSON text message."""
    return {
        "content": msg.content,
        "source_agent_id": msg.source_agent_id,
    }


# --- Integration tests ---


@pytest.fixture(scope="module")
def server():
    """Start a real AVP server on a free port for testing."""
    app = create_app(echo_handler, text_handler=text_handler)

    config = uvicorn.Config(app, host="127.0.0.1", port=9123, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(50):
        try:
            import httpx
            httpx.get("http://127.0.0.1:9123/health", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("Server did not start in time")

    yield "http://127.0.0.1:9123"

    server.should_exit = True
    thread.join(timeout=5)


def _make_metadata(emb, model_id="", agent_id=""):
    """Build AVPMetadata from a numpy embedding."""
    from avp.types import _STR_TO_DTYPE
    dtype_str = str(emb.dtype)
    return AVPMetadata(
        model_id=model_id,
        hidden_dim=emb.shape[0],
        payload_type=PayloadType.EMBEDDING,
        dtype=_STR_TO_DTYPE.get(dtype_str, DataType.FLOAT32),
        tensor_shape=emb.shape,
        source_agent_id=agent_id,
    )


def test_client_transmit(server):
    emb = np.random.randn(384).astype(np.float32)
    metadata = _make_metadata(emb, model_id="all-MiniLM-L6-v2", agent_id="test-agent")
    payload = embedding_to_bytes(emb)

    with AVPClient(server, agent_id="test-agent") as client:
        resp = client.transmit(payload, metadata)

    data = resp.json()
    assert data["success"] is True
    assert data["model_id"] == "all-MiniLM-L6-v2"
    assert data["embedding_dim"] == 384
    assert data["agent_id"] == "test-agent"
    assert data["compressed"] is False


def test_client_transmit_compressed(server):
    emb = np.random.randn(1024).astype(np.float32)
    metadata = _make_metadata(emb, model_id="test-model", agent_id="compress-agent")
    payload = embedding_to_bytes(emb)

    with AVPClient(server, agent_id="compress-agent") as client:
        resp = client.transmit(
            payload,
            metadata,
            compression=avp.CompressionLevel.BALANCED,
        )

    data = resp.json()
    assert data["success"] is True
    assert data["compressed"] is True
    assert data["embedding_dim"] == 1024


def test_client_transmit_float16(server):
    emb = np.random.randn(768).astype(np.float16)
    metadata = _make_metadata(emb, model_id="fp16-model", agent_id="fp16-agent")
    payload = embedding_to_bytes(emb)

    with AVPClient(server, agent_id="fp16-agent") as client:
        resp = client.transmit(payload, metadata)

    data = resp.json()
    assert data["success"] is True
    assert data["data_type"] == "float16"
    assert data["embedding_dim"] == 768


def test_client_send_text(server):
    msg = JSONMessage(
        session_id="test-session",
        source_agent_id="alice",
        target_agent_id="bob",
        content="Hello from fallback!",
    )

    with AVPClient(server, agent_id="alice") as client:
        resp = client.send_text(msg)

    data = resp.json()
    assert data["success"] is True
    assert data["content"] == "Hello from fallback!"
    assert data["source_agent_id"] == "alice"


def test_v2_transmit_rejects_bad_payload(server):
    """v2 endpoint should also reject garbage."""
    import httpx

    resp = httpx.post(
        f"{server}/avp/v2/transmit",
        content=b"garbage",
        headers={"Content-Type": "application/avp+binary"},
    )
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_health_endpoint(server):
    import httpx

    resp = httpx.get(f"{server}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# --- create_app unit test ---


def test_create_app_returns_fastapi():
    app = create_app(echo_handler)
    assert hasattr(app, "routes")
