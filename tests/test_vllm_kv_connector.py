"""Tests for AVPKVConnectorV1Dynamic (mock-based, no vLLM required)."""

import os

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


# ---------------------------------------------------------------------------
# Fixtures and mocks
# ---------------------------------------------------------------------------


class MockAttnMetadata:
    """Mock attention metadata with request_id."""

    def __init__(self, request_id="req-001"):
        self.request_id = request_id


class MockRequest:
    """Mock vLLM request object."""

    def __init__(self, request_id="req-001", prompt_token_ids=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids


class MockForwardContext:
    """Mock forward context."""

    def __init__(self, request_id="req-001"):
        self.request_id = request_id


class MockSchedulerOutput:
    """Mock scheduler output with scheduled_new_reqs."""

    def __init__(self, reqs=None):
        self.scheduled_new_reqs = reqs or []


@pytest.fixture
def store_dir(tmp_path):
    """Temporary store directory for KV-cache files."""
    return str(tmp_path / "avp_kv_store")


@pytest.fixture
def connector(store_dir):
    """Create an AVPKVConnectorV1Dynamic with temp store dir."""
    os.environ["AVP_KV_STORE_DIR"] = store_dir
    os.environ.pop("AVP_LATENT_STEPS", None)

    from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic

    class MockKVConfig:
        kv_connector_extra_config = {"avp_store_dir": store_dir, "avp_latent_steps": 0}

    class MockVLLMConfig:
        kv_transfer_config = MockKVConfig()

    return AVPKVConnectorV1Dynamic(vllm_config=MockVLLMConfig(), role=None)


# ---------------------------------------------------------------------------
# FileKVStore tests
# ---------------------------------------------------------------------------


class TestFileKVStore:
    def test_save_and_load_layer(self, store_dir):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(store_dir)
        tensor = torch.randn(2, 4, 8, 16)

        store.save_layer("test-key", 0, tensor)
        loaded = store.load_layer("test-key", 0)

        assert loaded is not None
        assert torch.allclose(loaded, tensor)

    def test_load_missing_returns_none(self, store_dir):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(store_dir)
        assert store.load_layer("nonexistent", 0) is None

    def test_has_key(self, store_dir):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(store_dir)
        assert store.has_key("missing") is False

        store.save_meta("present", seq_len=10, num_layers=2)
        assert store.has_key("present") is True

    def test_get_seq_len(self, store_dir):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(store_dir)
        store.save_meta("key1", seq_len=42, num_layers=4)
        assert store.get_seq_len("key1") == 42

    def test_get_seq_len_missing(self, store_dir):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(store_dir)
        assert store.get_seq_len("missing") == 0

    def test_delete(self, store_dir):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(store_dir)
        store.save_meta("to-delete", seq_len=5, num_layers=1)
        store.save_layer("to-delete", 0, torch.randn(2, 4, 8, 16))

        assert store.has_key("to-delete") is True
        store.delete("to-delete")
        assert store.has_key("to-delete") is False

    def test_get_num_layers(self, store_dir):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(store_dir)
        store.save_meta("key2", seq_len=10, num_layers=7)
        assert store.get_num_layers("key2") == 7


# ---------------------------------------------------------------------------
# Layout detection tests
# ---------------------------------------------------------------------------


class TestDetectKVLayout:
    def test_5d_stacked(self):
        from avp.connectors.vllm_kv_connector import _detect_kv_layout

        t = torch.randn(1, 2, 4, 8, 16)
        assert _detect_kv_layout(t) == "stacked_5d"

    def test_4d_stacked(self):
        from avp.connectors.vllm_kv_connector import _detect_kv_layout

        t = torch.randn(2, 4, 8, 16)
        assert _detect_kv_layout(t) == "stacked_4d"

    def test_unknown_layout(self):
        from avp.connectors.vllm_kv_connector import _detect_kv_layout

        t = torch.randn(3, 4, 8)
        assert _detect_kv_layout(t) == "unknown"


class TestExtractKV:
    def test_extract_5d_single_batch(self):
        from avp.connectors.vllm_kv_connector import _extract_kv_from_layer

        t = torch.randn(1, 2, 4, 8, 16)
        k, v = _extract_kv_from_layer(t)

        assert k.shape == (4, 8, 16)
        assert v.shape == (4, 8, 16)
        assert torch.allclose(k, t[0, 0])
        assert torch.allclose(v, t[0, 1])

    def test_extract_4d(self):
        from avp.connectors.vllm_kv_connector import _extract_kv_from_layer

        t = torch.randn(2, 4, 8, 16)
        k, v = _extract_kv_from_layer(t)

        assert k.shape == (4, 8, 16)
        assert v.shape == (4, 8, 16)
        assert torch.allclose(k, t[0])
        assert torch.allclose(v, t[1])

    def test_extract_unknown_raises(self):
        from avp.connectors.vllm_kv_connector import _extract_kv_from_layer

        t = torch.randn(3, 4, 8)
        with pytest.raises(ValueError, match="Unrecognized"):
            _extract_kv_from_layer(t)


# ---------------------------------------------------------------------------
# Connector tests
# ---------------------------------------------------------------------------


class TestConnectorInit:
    def test_creates_store(self, connector, store_dir):
        assert os.path.isdir(store_dir)

    def test_role_defaults_to_worker(self, connector):
        from avp.connectors.vllm_kv_connector import KVConnectorRole

        assert connector.role == KVConnectorRole.WORKER

    def test_requires_piecewise(self, connector):
        assert connector.requires_piecewise_for_cudagraph() is True

    def test_latent_steps_env_var_set(self, store_dir):
        """Connector bridges latent_steps config to env var."""
        os.environ.pop("AVP_LATENT_STEPS", None)

        from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic

        class MockKVConfig:
            kv_connector_extra_config = {"avp_store_dir": store_dir, "avp_latent_steps": 7}

        class MockVLLMConfig:
            kv_transfer_config = MockKVConfig()

        AVPKVConnectorV1Dynamic(vllm_config=MockVLLMConfig(), role=None)
        assert os.environ.get("AVP_LATENT_STEPS") == "7"


class TestSaveKVLayer:
    def test_buffers_tensors(self, connector):
        meta = MockAttnMetadata("req-001")
        t = torch.randn(1, 2, 4, 8, 16)

        connector.save_kv_layer("model.layers.0.self_attn", t, meta)
        connector.save_kv_layer("model.layers.1.self_attn", t, meta)

        assert "req-001" in connector._pending_saves
        assert len(connector._pending_saves["req-001"]) == 2

    def test_tracks_seq_len(self, connector):
        meta = MockAttnMetadata("req-002")
        t = torch.randn(1, 2, 4, 8, 16)  # seq_len=8

        connector.save_kv_layer("model.layers.0.self_attn", t, meta)

        assert connector._pending_meta.get("req-002") == 8


class TestFlushAndSave:
    def test_wait_for_save_flushes_to_store(self, connector):
        meta = MockAttnMetadata("req-003")
        t = torch.randn(1, 2, 4, 8, 16)

        connector.save_kv_layer("model.layers.0.self_attn", t, meta)
        connector.save_kv_layer("model.layers.1.self_attn", t, meta)
        connector.wait_for_save()

        # Buffer should be cleared
        assert "req-003" not in connector._pending_saves

        # Store should have data
        assert connector._store.has_key("req-003")
        assert connector._store.get_seq_len("req-003") == 8

    def test_request_finished_triggers_flush(self, connector):
        meta = MockAttnMetadata("req-004")
        t = torch.randn(1, 2, 4, 8, 16)

        connector.save_kv_layer("model.layers.0.self_attn", t, meta)
        connector.request_finished(MockRequest("req-004"))

        assert connector._store.has_key("req-004")


class TestLoadKV:
    def test_start_load_finds_data(self, connector):
        # First, produce data
        meta = MockAttnMetadata("req-005")
        t = torch.randn(1, 2, 4, 8, 16)
        connector.save_kv_layer("model.layers.0.self_attn", t, meta)
        connector.wait_for_save()

        # Load
        ctx = MockForwardContext("req-005")
        connector.start_load_kv(ctx)

        assert "req-005" in connector._loaded_keys

    def test_start_load_no_data(self, connector):
        ctx = MockForwardContext("nonexistent")
        connector.start_load_kv(ctx)

        assert "nonexistent" not in connector._loaded_keys

    def test_start_load_none_context(self, connector):
        connector.start_load_kv(None)
        assert len(connector._loaded_keys) == 0

    def test_wait_for_layer_load_returns_tensor(self, connector):
        # Produce and flush
        meta = MockAttnMetadata("req-006")
        t = torch.randn(1, 2, 4, 8, 16)
        connector.save_kv_layer("model.layers.0.self_attn", t, meta)
        connector.wait_for_save()

        # Load
        ctx = MockForwardContext("req-006")
        connector.start_load_kv(ctx)

        result = connector.wait_for_layer_load("model.layers.0.self_attn")
        assert result is not None
        # Should be [2, num_kv_heads, seq_len, head_dim]
        assert result.shape[0] == 2

    def test_wait_for_layer_load_none_when_empty(self, connector):
        result = connector.wait_for_layer_load("model.layers.0.self_attn")
        assert result is None


class TestRoundTrip:
    def test_save_load_roundtrip(self, connector):
        """Full round-trip: save layers -> flush -> load -> verify."""
        num_kv_heads, seq_len, head_dim = 4, 16, 8
        meta = MockAttnMetadata("roundtrip-001")

        layer0 = torch.randn(1, 2, num_kv_heads, seq_len, head_dim)
        layer1 = torch.randn(1, 2, num_kv_heads, seq_len, head_dim)

        connector.save_kv_layer("model.layers.0.self_attn", layer0, meta)
        connector.save_kv_layer("model.layers.1.self_attn", layer1, meta)
        connector.wait_for_save()

        # Load
        ctx = MockForwardContext("roundtrip-001")
        connector.start_load_kv(ctx)

        # Verify both layers
        r0 = connector.wait_for_layer_load("model.layers.0.self_attn")
        r1 = connector.wait_for_layer_load("model.layers.1.self_attn")

        assert r0 is not None
        assert r1 is not None

        # Verify shapes: [2, num_kv_heads, seq_len, head_dim]
        assert r0.shape == (2, num_kv_heads, seq_len, head_dim)
        assert r1.shape == (2, num_kv_heads, seq_len, head_dim)

        # Verify data integrity
        k0_orig, v0_orig = layer0[0, 0], layer0[0, 1]
        assert torch.allclose(r0[0], k0_orig, atol=1e-6)
        assert torch.allclose(r0[1], v0_orig, atol=1e-6)


class TestSchedulerMethods:
    def test_get_num_new_matched_tokens_with_data(self, connector):
        # get_num_new_matched_tokens uses AVPReqMeta.from_request() which
        # derives store_key from prompt_token_ids via hash. We need the
        # store data to be saved under that same key.
        from avp.connectors.vllm_kv_connector import compute_request_hash

        prompt_ids = [1, 2, 3]
        store_key = compute_request_hash(prompt_ids)

        # Save data under the hash-derived key
        meta = MockAttnMetadata(store_key)
        t = torch.randn(1, 2, 4, 20, 16)  # seq_len=20
        connector.save_kv_layer("model.layers.0.self_attn", t, meta)
        connector.wait_for_save()

        req = MockRequest("req-sched", prompt_token_ids=prompt_ids)
        matched, is_async = connector.get_num_new_matched_tokens(req, num_computed_tokens=5)

        assert matched == 15  # 20 - 5
        assert is_async is False

    def test_get_num_new_matched_tokens_no_data(self, connector):
        req = MockRequest("nonexistent", prompt_token_ids=[1, 2, 3])
        matched, is_async = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)

        assert matched == 0
        assert is_async is False

    def test_build_connector_meta(self, connector):
        # Save some data first
        meta = MockAttnMetadata("req-meta")
        t = torch.randn(1, 2, 4, 8, 16)
        connector.save_kv_layer("model.layers.0.self_attn", t, meta)
        connector.wait_for_save()

        req = MockRequest("req-meta", prompt_token_ids=[10, 20, 30])
        scheduler_output = MockSchedulerOutput(reqs=[req])

        result = connector.build_connector_meta(scheduler_output)

        assert hasattr(result, "requests")

    def test_update_state_after_alloc_noop(self, connector):
        """update_state_after_alloc with 0 external tokens is a no-op."""
        req = MockRequest("req-alloc")
        connector.update_state_after_alloc(req, blocks=[[1, 2, 3]], num_external_tokens=0)
        # Should not raise


class TestRequestFinished:
    def test_returns_false_none(self, connector):
        """request_finished returns (False, None) to keep blocks."""
        result = connector.request_finished(MockRequest("req-fin"))
        assert result == (False, None)

    def test_cleans_up_loaded_state(self, connector):
        # Simulate loaded state
        connector._loaded_keys["req-cleanup"] = "key-cleanup"

        connector.request_finished(MockRequest("req-cleanup"))

        assert "req-cleanup" not in connector._loaded_keys


class TestStats:
    def test_get_stats(self, connector):
        stats = connector.get_kv_connector_stats()
        assert "pending_saves" in stats
        assert "loaded_requests" in stats
        assert "store_dir" in stats
        assert stats["pending_saves"] == 0
        assert stats["loaded_requests"] == 0


class TestStubMethods:
    def test_handle_preemptions_noop(self, connector):
        connector.handle_preemptions()

    def test_get_block_ids_with_load_errors_empty(self, connector):
        assert connector.get_block_ids_with_load_errors() == set()

    def test_register_kv_caches(self, connector):
        caches = {"layer0": torch.zeros(10), "layer1": torch.zeros(10)}
        connector.register_kv_caches(caches)
        assert connector._kv_caches is not None
        assert len(connector._kv_caches) == 2


class TestMultipleRequests:
    def test_isolated_requests(self, connector):
        meta_a = MockAttnMetadata("req-a")
        meta_b = MockAttnMetadata("req-b")
        ta = torch.randn(1, 2, 4, 8, 16)
        tb = torch.randn(1, 2, 4, 8, 16)

        connector.save_kv_layer("model.layers.0.self_attn", ta, meta_a)
        connector.save_kv_layer("model.layers.0.self_attn", tb, meta_b)

        assert len(connector._pending_saves) == 2
        assert "req-a" in connector._pending_saves
        assert "req-b" in connector._pending_saves


# ---------------------------------------------------------------------------
# Module-level helper tests
# ---------------------------------------------------------------------------


class TestParseLayerIndex:
    def test_standard_names(self):
        from avp.connectors.vllm_kv_connector import _parse_layer_index

        assert _parse_layer_index("model.layers.5.self_attn") == 5
        assert _parse_layer_index("model.layers.0.self_attn") == 0
        assert _parse_layer_index("model.layers.31.self_attn") == 31

    def test_no_match(self):
        from avp.connectors.vllm_kv_connector import _parse_layer_index

        assert _parse_layer_index("no_layer_here") is None


class TestComputeRequestHash:
    def test_deterministic(self):
        from avp.connectors.vllm_kv_connector import compute_request_hash

        h1 = compute_request_hash([1, 2, 3])
        h2 = compute_request_hash([1, 2, 3])
        assert h1 == h2

    def test_different_inputs(self):
        from avp.connectors.vllm_kv_connector import compute_request_hash

        h1 = compute_request_hash([1, 2, 3])
        h2 = compute_request_hash([4, 5, 6])
        assert h1 != h2

    def test_length(self):
        from avp.connectors.vllm_kv_connector import compute_request_hash

        h = compute_request_hash([1, 2, 3])
        assert len(h) == 16


class TestExtractRequestId:
    def test_from_attn_metadata(self, connector):
        assert connector._extract_request_id(MockAttnMetadata("req-x")) == "req-x"

    def test_from_request(self, connector):
        assert connector._extract_request_id(MockRequest("req-y")) == "req-y"

    def test_from_forward_context(self, connector):
        assert connector._extract_request_id(MockForwardContext("req-z")) == "req-z"

    def test_from_string(self, connector):
        assert connector._extract_request_id("direct-id") == "direct-id"

    def test_fallback_default(self, connector):
        assert connector._extract_request_id(object()) == "default"

    def test_from_list(self, connector):
        """DBO mode passes attn_metadata as list."""
        items = [MockAttnMetadata("req-dbo")]
        assert connector._extract_request_id(items) == "req-dbo"


class TestDeriveStoreKey:
    def test_uses_prompt_token_ids_when_available(self, connector):
        """Store key is derived from prompt tokens, not request_id."""
        from avp.connectors.vllm_kv_connector import compute_request_hash

        req = MockRequest("req-123", prompt_token_ids=[10, 20, 30])
        key = connector._derive_store_key(req)
        assert key == compute_request_hash([10, 20, 30])
        assert key != "req-123"

    def test_falls_back_to_request_id(self, connector):
        """Without prompt_token_ids, falls back to request_id."""
        req = MockRequest("req-fallback")
        key = connector._derive_store_key(req)
        assert key == "req-fallback"

    def test_producer_consumer_keys_match(self, connector):
        """Producer and consumer derive the same key for same prompt."""
        from avp.connectors.vllm_kv_connector import compute_request_hash

        prompt_ids = [1, 2, 3, 4, 5]
        expected_key = compute_request_hash(prompt_ids)

        # Producer side (attn_metadata with prompt_token_ids)
        producer_meta = MockRequest("req-prod", prompt_token_ids=prompt_ids)
        producer_key = connector._derive_store_key(producer_meta)

        # Consumer side (request with same prompt_token_ids)
        consumer_req = MockRequest("req-cons", prompt_token_ids=prompt_ids)
        consumer_key = connector._derive_store_key(consumer_req)

        assert producer_key == expected_key
        assert consumer_key == expected_key
        assert producer_key == consumer_key


class TestMultiBlockExtraction:
    def test_multi_block_5d_data_integrity(self):
        """Multi-block extraction preserves data correctly."""
        from avp.connectors.vllm_kv_connector import _extract_kv_from_layer

        # 3 blocks, 2 (K/V), 4 heads, 8 tokens per block, 16 head_dim
        num_blocks, num_heads, tokens_per_block, head_dim = 3, 4, 8, 16
        t = torch.randn(num_blocks, 2, num_heads, tokens_per_block, head_dim)

        k, v = _extract_kv_from_layer(t)

        # Shape: [num_heads, total_tokens, head_dim]
        assert k.shape == (num_heads, num_blocks * tokens_per_block, head_dim)
        assert v.shape == (num_heads, num_blocks * tokens_per_block, head_dim)

        # Verify data from each block is in the correct position
        for block_idx in range(num_blocks):
            start = block_idx * tokens_per_block
            end = start + tokens_per_block
            assert torch.allclose(k[:, start:end, :], t[block_idx, 0])
            assert torch.allclose(v[:, start:end, :], t[block_idx, 1])
