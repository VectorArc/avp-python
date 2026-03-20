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


class MockRequest:
    def __init__(self, request_id="req-001", prompt_token_ids=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids


class MockForwardContext:
    def __init__(self, request_id="req-001"):
        self.request_id = request_id


@pytest.fixture
def store_dir(tmp_path):
    return str(tmp_path / "avp_kv_store")


@pytest.fixture
def connector(store_dir):
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
        store.save_layer("to-delete", 0, torch.randn(2, 4, 8))
        assert store.has_key("to-delete") is True
        store.delete("to-delete")
        assert store.has_key("to-delete") is False

    def test_get_num_layers(self, store_dir):
        from avp.connectors.vllm_kv_connector import FileKVStore
        store = FileKVStore(store_dir)
        store.save_meta("key2", seq_len=10, num_layers=7)
        assert store.get_num_layers("key2") == 7


# ---------------------------------------------------------------------------
# Slot mapping tests
# ---------------------------------------------------------------------------


class TestSlotMapping:
    def test_compute_slot_mapping(self):
        from avp.connectors.vllm_kv_connector import _compute_slot_mapping

        # 3 blocks, block_size=4, 10 tokens
        mapping = _compute_slot_mapping([2, 5, 8], block_size=4, num_tokens=10)

        # Block 2: slots 8,9,10,11
        # Block 5: slots 20,21,22,23
        # Block 8: slots 32,33,34,35 (but only 2 used: 32,33)
        assert mapping.shape == (10,)
        assert mapping[0].item() == 8   # block 2, offset 0
        assert mapping[4].item() == 20  # block 5, offset 0
        assert mapping[9].item() == 33  # block 8, offset 1

    def test_extract_request_kv(self):
        from avp.connectors.vllm_kv_connector import _extract_request_kv

        # Full paged buffer: [2, 10 blocks, 4 block_size, 2 kv_heads, 8 head_dim]
        kv_cache = torch.randn(2, 10, 4, 2, 8)
        slot_mapping = torch.tensor([0, 1, 4, 5])  # block 0 slots 0,1 + block 1 slots 0,1

        extracted = _extract_request_kv(kv_cache, slot_mapping)

        # Shape: [2, 4 tokens, 2*8=16 features]
        assert extracted.shape == (2, 4, 16)

    def test_inject_request_kv(self):
        from avp.connectors.vllm_kv_connector import _inject_request_kv

        kv_cache = torch.zeros(2, 10, 4, 2, 8)
        slot_mapping = torch.tensor([8, 9])  # block 2, offsets 0,1
        kv_data = torch.ones(2, 2, 16)

        _inject_request_kv(kv_cache, slot_mapping, kv_data)

        # Verify data was injected at the right slots
        flat = kv_cache.reshape(2, 40, -1)
        assert torch.allclose(flat[:, 8, :], torch.ones(2, 16))
        assert torch.allclose(flat[:, 9, :], torch.ones(2, 16))
        # Other slots should still be zero
        assert flat[:, 0, :].sum().item() == 0

    def test_extract_inject_roundtrip(self):
        from avp.connectors.vllm_kv_connector import (
            _extract_request_kv, _inject_request_kv,
        )

        # Create a buffer with known data at specific slots
        kv_cache = torch.randn(2, 10, 4, 2, 8)
        slot_mapping = torch.tensor([4, 5, 6, 7, 12, 13])  # block 1 + half of block 3

        # Extract
        extracted = _extract_request_kv(kv_cache, slot_mapping)

        # Inject into a fresh buffer
        fresh_cache = torch.zeros_like(kv_cache)
        _inject_request_kv(fresh_cache, slot_mapping, extracted)

        # Verify roundtrip preserves data at the mapped slots
        flat_orig = kv_cache.reshape(2, 40, -1)
        flat_fresh = fresh_cache.reshape(2, 40, -1)
        for slot in slot_mapping.tolist():
            assert torch.allclose(flat_orig[:, slot, :], flat_fresh[:, slot, :])


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
        os.environ.pop("AVP_LATENT_STEPS", None)
        from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic

        class MockKVConfig:
            kv_connector_extra_config = {"avp_store_dir": store_dir, "avp_latent_steps": 7}

        class MockVLLMConfig:
            kv_transfer_config = MockKVConfig()

        AVPKVConnectorV1Dynamic(vllm_config=MockVLLMConfig(), role=None)
        assert os.environ.get("AVP_LATENT_STEPS") == "7"


class TestRequestFinished:
    def test_extracts_kv_when_caches_registered(self, connector):
        """request_finished extracts KV from registered GPU buffers."""
        block_size = connector._block_size  # 16
        num_blocks = 20
        num_kv_heads, head_dim = 2, 8

        kv_layer_0 = torch.randn(2, num_blocks, block_size, num_kv_heads, head_dim)
        kv_layer_1 = torch.randn(2, num_blocks, block_size, num_kv_heads, head_dim)
        connector._kv_caches = {
            "model.layers.0.self_attn": kv_layer_0,
            "model.layers.1.self_attn": kv_layer_1,
        }

        req = MockRequest("req-001", prompt_token_ids=[1, 2, 3, 4, 5])
        block_ids = [2, 3]  # 2 blocks for 5 tokens (block_size=16, ceil(5/16)=1, but 2 blocks allocated)

        result = connector.request_finished(req, block_ids=block_ids)

        from avp.connectors.vllm_kv_connector import compute_request_hash
        store_key = compute_request_hash([1, 2, 3, 4, 5])
        assert connector._store.has_key(store_key)
        assert connector._store.get_seq_len(store_key) == 5

        layer_data = connector._store.load_layer(store_key, 0)
        assert layer_data is not None
        assert layer_data.shape[1] == 5  # 5 tokens

    def test_no_extraction_without_caches(self, connector):
        """request_finished does nothing if kv_caches not registered."""
        req = MockRequest("req-002", prompt_token_ids=[1, 2, 3])
        result = connector.request_finished(req, block_ids=[0])
        assert result == (True, None)

    def test_no_extraction_without_block_ids(self, connector):
        connector._kv_caches = {"layer.0": torch.randn(2, 10, 4, 2, 8)}
        req = MockRequest("req-003", prompt_token_ids=[1, 2])
        result = connector.request_finished(req, block_ids=None)
        assert result == (True, None)


class TestGetNumNewMatchedTokens:
    def test_returns_stored_count(self, connector):
        from avp.connectors.vllm_kv_connector import compute_request_hash
        prompt_ids = [10, 20, 30]
        store_key = compute_request_hash(prompt_ids)
        connector._store.save_meta(store_key, seq_len=50, num_layers=2)

        req = MockRequest("req-match", prompt_token_ids=prompt_ids)
        matched, is_async = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
        assert matched == 50
        assert is_async is False

    def test_returns_zero_when_not_stored(self, connector):
        req = MockRequest("req-none", prompt_token_ids=[99, 98])
        matched, is_async = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
        assert matched == 0

    def test_subtracts_computed_tokens(self, connector):
        from avp.connectors.vllm_kv_connector import compute_request_hash
        prompt_ids = [1, 2, 3]
        store_key = compute_request_hash(prompt_ids)
        connector._store.save_meta(store_key, seq_len=30, num_layers=1)

        req = MockRequest("req-partial", prompt_token_ids=prompt_ids)
        matched, _ = connector.get_num_new_matched_tokens(req, num_computed_tokens=10)
        assert matched == 20


class TestUpdateStateAfterAlloc:
    def test_records_pending_load(self, connector):
        req = MockRequest("req-alloc", prompt_token_ids=[1, 2])

        class MockBlocks:
            def get_block_ids(self):
                return ([5, 6, 7],)

        connector.update_state_after_alloc(req, MockBlocks(), num_external_tokens=10)
        assert "req-alloc" in connector._pending_loads
        assert connector._pending_loads["req-alloc"].block_ids == [5, 6, 7]
        assert connector._pending_loads["req-alloc"].num_external_tokens == 10

    def test_ignores_zero_external(self, connector):
        req = MockRequest("req-zero")
        connector.update_state_after_alloc(req, None, num_external_tokens=0)
        assert "req-zero" not in connector._pending_loads


class TestStartLoadKV:
    def test_injects_stored_kv(self, connector):
        """Full roundtrip: save via request_finished, load via start_load_kv."""
        # Use block_size=16 (connector default) with enough blocks
        block_size = connector._block_size  # 16
        num_blocks = 20
        num_kv_heads, head_dim = 2, 8

        kv_0 = torch.randn(2, num_blocks, block_size, num_kv_heads, head_dim)
        kv_1 = torch.randn(2, num_blocks, block_size, num_kv_heads, head_dim)
        connector._kv_caches = {
            "model.layers.0.self_attn": kv_0,
            "model.layers.1.self_attn": kv_1,
        }

        # Agent A: save KV (4 tokens → 1 block)
        prompt_ids = [1, 2, 3, 4]
        req_a = MockRequest("req-a", prompt_token_ids=prompt_ids)
        connector.request_finished(req_a, block_ids=[2])

        from avp.connectors.vllm_kv_connector import compute_request_hash
        store_key = compute_request_hash(prompt_ids)
        assert connector._store.has_key(store_key)

        # Agent B: inject into fresh buffer
        fresh_kv_0 = torch.zeros_like(kv_0)
        fresh_kv_1 = torch.zeros_like(kv_1)
        connector._kv_caches = {
            "model.layers.0.self_attn": fresh_kv_0,
            "model.layers.1.self_attn": fresh_kv_1,
        }

        from avp.connectors.vllm_kv_connector import AVPReqMeta
        meta_b = AVPReqMeta(
            request_id="req-b", store_key=store_key,
            num_external_tokens=4, block_ids=[5],
        )
        connector._pending_loads["req-b"] = meta_b

        connector.start_load_kv(forward_context=MockForwardContext())

        # Verify injection at block 5's slots (slot 80 = block 5 * 16)
        total_slots = num_blocks * block_size
        flat = fresh_kv_0.reshape(2, total_slots, -1)
        slot_80 = 5 * block_size  # = 80
        assert flat[:, slot_80, :].abs().sum().item() > 0


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

    def test_get_stats_returns_none(self, connector):
        assert connector.get_kv_connector_stats() is None

    def test_save_kv_layer_noop(self, connector):
        """save_kv_layer is a no-op (extraction happens in request_finished)."""
        connector.save_kv_layer("model.layers.0", torch.zeros(1), None)


# ---------------------------------------------------------------------------
# Module-level helpers
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
        assert compute_request_hash([1, 2, 3]) == compute_request_hash([1, 2, 3])

    def test_different_inputs(self):
        from avp.connectors.vllm_kv_connector import compute_request_hash
        assert compute_request_hash([1, 2, 3]) != compute_request_hash([4, 5, 6])

    def test_length(self):
        from avp.connectors.vllm_kv_connector import compute_request_hash
        assert len(compute_request_hash([1, 2, 3])) == 16
