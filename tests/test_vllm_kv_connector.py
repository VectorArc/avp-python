"""Tests for AVPKVConnectorV1Dynamic (mock-based, no vLLM required)."""

import os
import tempfile

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture
def store_dir(tmp_path):
    """Temporary store directory for KV-cache files."""
    return str(tmp_path / "avp_kv_store")


@pytest.fixture
def connector(store_dir, monkeypatch):
    """Create an AVPKVConnectorV1Dynamic with temp store dir."""
    monkeypatch.setenv("AVP_KV_STORE_DIR", store_dir)
    monkeypatch.setenv("AVP_NUM_LAYERS", "2")
    monkeypatch.setenv("AVP_BLOCK_SIZE", "4")

    from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic
    return AVPKVConnectorV1Dynamic()


class MockAttnMetadata:
    """Mock attention metadata with request_id."""
    def __init__(self, request_id="req-001"):
        self.request_id = request_id


class MockRequest:
    """Mock vLLM request object."""
    def __init__(self, request_id="req-001"):
        self.request_id = request_id


class MockForwardContext:
    """Mock forward context."""
    def __init__(self, request_id="req-001"):
        self.request_id = request_id


def test_init_creates_store_dir(connector, store_dir):
    """Connector creates store directory on init."""
    assert os.path.isdir(store_dir)


def test_role_is_worker(connector):
    from avp.connectors.vllm_kv_connector import KVConnectorRole
    assert connector.role == KVConnectorRole.WORKER


def test_save_kv_layer_buffers_tensors(connector):
    """save_kv_layer stores tensors in the buffer."""
    meta = MockAttnMetadata("req-001")
    t = torch.randn(1, 2, 4, 8, 16)  # [batch, 2, heads, tokens, head_dim]

    connector.save_kv_layer("model.layers.0.self_attn", t, meta)
    connector.save_kv_layer("model.layers.1.self_attn", t, meta)

    assert "req-001" in connector._save_buffers
    assert len(connector._save_buffers["req-001"].tensors) == 2


def test_wait_for_save_serializes_to_store(connector, store_dir):
    """wait_for_save serializes buffered layers and writes to store."""
    meta = MockAttnMetadata("req-002")
    t = torch.randn(1, 2, 4, 8, 16)

    connector.save_kv_layer("model.layers.0.self_attn", t, meta)
    connector.save_kv_layer("model.layers.1.self_attn", t, meta)
    connector.wait_for_save()

    # Buffer should be cleared
    assert "req-002" not in connector._save_buffers

    # Store file should exist
    store_path = os.path.join(store_dir, "req-002.avp")
    assert os.path.exists(store_path)
    assert os.path.getsize(store_path) > 0


def test_request_finished_triggers_flush(connector, store_dir):
    """request_finished triggers serialization of pending buffer."""
    meta = MockAttnMetadata("req-003")
    t = torch.randn(1, 2, 4, 8, 16)

    connector.save_kv_layer("model.layers.0.self_attn", t, meta)
    connector.request_finished(MockRequest("req-003"))

    store_path = os.path.join(store_dir, "req-003.avp")
    assert os.path.exists(store_path)


def test_start_load_kv_reads_from_store(connector, store_dir):
    """start_load_kv reads KV-cache from store into loaded buffer."""
    # First, produce a store file
    meta = MockAttnMetadata("req-004")
    t = torch.randn(1, 2, 4, 8, 16)
    connector.save_kv_layer("model.layers.0.self_attn", t, meta)
    connector.save_kv_layer("model.layers.1.self_attn", t, meta)
    connector.wait_for_save()

    # Now load
    ctx = MockForwardContext("req-004")
    connector.start_load_kv(ctx)

    assert "req-004" in connector._loaded_kv
    assert len(connector._loaded_kv["req-004"]) == 2


def test_wait_for_layer_load_returns_tensor(connector):
    """wait_for_layer_load returns KV data for a loaded layer."""
    # Manually populate loaded data
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    connector._loaded_kv["req-005"] = [(k, v)]

    result = connector.wait_for_layer_load("model.layers.0.self_attn")
    assert result is not None
    # Should be [2, 4, 8, 16] (K and V concatenated along dim 0)
    assert result.shape[0] == 2


def test_wait_for_layer_load_returns_none_when_empty(connector):
    """wait_for_layer_load returns None when no data loaded."""
    result = connector.wait_for_layer_load("model.layers.0.self_attn")
    assert result is None


def test_get_num_new_matched_tokens(connector, store_dir):
    """get_num_new_matched_tokens returns correct token count from store."""
    from avp.kv_cache import KVCacheHeader

    # Write a minimal store file with known seq_len
    header = KVCacheHeader(num_layers=2, num_kv_heads=4, head_dim=16, seq_len=100, dtype="float16")
    data = header.to_bytes()
    # Pad with enough bytes for 2 layers of KV data
    tensor_bytes = 4 * 100 * 16 * 2  # num_kv_heads * seq_len * head_dim * bytes_per_element
    data += b"\x00" * (2 * 2 * tensor_bytes)  # 2 layers * 2 (K+V) * tensor_bytes

    os.makedirs(store_dir, exist_ok=True)
    store_path = os.path.join(store_dir, "req-006.avp")
    with open(store_path, "wb") as f:
        f.write(data)

    matched, is_async = connector.get_num_new_matched_tokens(MockRequest("req-006"), num_computed_tokens=30)
    assert matched == 70  # 100 - 30
    assert is_async is False


def test_get_num_new_matched_tokens_no_store(connector):
    """get_num_new_matched_tokens returns (0, False) when no store file exists."""
    matched, is_async = connector.get_num_new_matched_tokens(MockRequest("nonexistent"), num_computed_tokens=0)
    assert matched == 0
    assert is_async is False


def test_roundtrip_save_load(connector):
    """Full round-trip: save layers → flush → load → verify layer data."""
    num_kv_heads, seq_len, head_dim = 4, 16, 8
    meta = MockAttnMetadata("roundtrip-001")

    # Create contiguous KV tensors (batch=1 format, as if from a single sequence)
    layer0 = torch.randn(1, 2, num_kv_heads, seq_len, head_dim)
    layer1 = torch.randn(1, 2, num_kv_heads, seq_len, head_dim)

    connector.save_kv_layer("model.layers.0.self_attn", layer0, meta)
    connector.save_kv_layer("model.layers.1.self_attn", layer1, meta)
    connector.wait_for_save()

    # Load
    ctx = MockForwardContext("roundtrip-001")
    connector.start_load_kv(ctx)

    # Verify layer 0
    result0 = connector.wait_for_layer_load("model.layers.0.self_attn")
    assert result0 is not None

    # Verify layer 1
    result1 = connector.wait_for_layer_load("model.layers.1.self_attn")
    assert result1 is not None


def test_register_kv_caches(connector):
    """register_kv_caches stores reference."""
    caches = {"layer0": torch.zeros(10), "layer1": torch.zeros(10)}
    connector.register_kv_caches(caches)
    assert connector._kv_caches is not None
    assert len(connector._kv_caches) == 2


def test_get_kv_connector_stats(connector):
    """get_kv_connector_stats returns expected keys."""
    stats = connector.get_kv_connector_stats()
    assert "pending_saves" in stats
    assert "loaded_requests" in stats
    assert "store_dir" in stats
    assert stats["pending_saves"] == 0


def test_handle_preemptions_noop(connector):
    """handle_preemptions is a no-op and doesn't raise."""
    connector.handle_preemptions()


def test_get_block_ids_with_load_errors_empty(connector):
    """get_block_ids_with_load_errors returns empty set."""
    assert connector.get_block_ids_with_load_errors() == set()


def test_parse_layer_index():
    from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic

    assert AVPKVConnectorV1Dynamic._parse_layer_index("model.layers.5.self_attn") == 5
    assert AVPKVConnectorV1Dynamic._parse_layer_index("model.layers.0.self_attn") == 0
    assert AVPKVConnectorV1Dynamic._parse_layer_index("model.layers.31.self_attn") == 31
    assert AVPKVConnectorV1Dynamic._parse_layer_index("no_layer_here") is None


def test_compute_request_hash():
    from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic

    h1 = AVPKVConnectorV1Dynamic.compute_request_hash([1, 2, 3])
    h2 = AVPKVConnectorV1Dynamic.compute_request_hash([1, 2, 3])
    h3 = AVPKVConnectorV1Dynamic.compute_request_hash([4, 5, 6])

    assert h1 == h2  # deterministic
    assert h1 != h3  # different inputs
    assert len(h1) == 16  # truncated hex


def test_start_load_kv_no_context(connector):
    """start_load_kv with None context is a no-op."""
    connector.start_load_kv(None)
    assert len(connector._loaded_kv) == 0


def test_multiple_requests_isolated(connector):
    """Multiple concurrent requests don't interfere."""
    meta_a = MockAttnMetadata("req-a")
    meta_b = MockAttnMetadata("req-b")
    ta = torch.randn(1, 2, 4, 8, 16)
    tb = torch.randn(1, 2, 4, 8, 16)

    connector.save_kv_layer("model.layers.0.self_attn", ta, meta_a)
    connector.save_kv_layer("model.layers.0.self_attn", tb, meta_b)

    assert len(connector._save_buffers) == 2
    assert "req-a" in connector._save_buffers
    assert "req-b" in connector._save_buffers
