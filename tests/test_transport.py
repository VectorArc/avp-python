"""Tests for AVP transport layer (client + server)."""

import asyncio
import threading
import time

import numpy as np
import pytest
import uvicorn

import avp
from avp.transport import AVPClient, create_app


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


# --- Integration tests ---


@pytest.fixture(scope="module")
def server():
    """Start a real AVP server on a free port for testing."""
    app = create_app(echo_handler)

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


def test_client_transmit(server):
    emb = np.random.randn(384).astype(np.float32)

    with AVPClient(server, agent_id="test-agent") as client:
        resp = client.transmit(emb, model_id="all-MiniLM-L6-v2")

    data = resp.json()
    assert data["success"] is True
    assert data["model_id"] == "all-MiniLM-L6-v2"
    assert data["embedding_dim"] == 384
    assert data["agent_id"] == "test-agent"
    assert data["compressed"] is False


def test_client_transmit_compressed(server):
    emb = np.random.randn(1024).astype(np.float32)

    with AVPClient(server, agent_id="compress-agent") as client:
        resp = client.transmit(
            emb,
            model_id="test-model",
            compression=avp.CompressionLevel.BALANCED,
        )

    data = resp.json()
    assert data["success"] is True
    assert data["compressed"] is True
    assert data["embedding_dim"] == 1024


def test_client_transmit_float16(server):
    emb = np.random.randn(768).astype(np.float16)

    with AVPClient(server, agent_id="fp16-agent") as client:
        resp = client.transmit(emb, model_id="fp16-model")

    data = resp.json()
    assert data["success"] is True
    assert data["data_type"] == "float16"
    assert data["embedding_dim"] == 768


def test_server_rejects_bad_payload(server):
    """Sending garbage bytes should return 400."""
    import httpx

    resp = httpx.post(
        f"{server}/avp/v1/transmit",
        content=b"not an avp message",
        headers={"Content-Type": "application/avp+binary"},
    )
    data = resp.json()
    assert data["success"] is False
    assert resp.status_code == 400


def test_health_endpoint(server):
    import httpx

    resp = httpx.get(f"{server}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# --- create_app unit test ---


def test_create_app_returns_fastapi():
    app = create_app(echo_handler)
    assert hasattr(app, "routes")
