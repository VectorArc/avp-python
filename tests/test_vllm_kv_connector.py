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
    def test_flushes_pending_saves(self, connector):
        """request_finished triggers flush of pending layer data."""
        # Manually populate pending saves (simulating save_kv_layer)
        connector._pending_saves["test-key"] = {
            "num_tokens": 5,
            0: torch.randn(2, 5, 16),
            1: torch.randn(2, 5, 16),
        }

        req = MockRequest("req-001")
        connector.request_finished(req)

        # Pending should be flushed
        assert len(connector._pending_saves) == 0
        assert connector._store.has_key("test-key")
        assert connector._store.get_seq_len("test-key") == 5

    def test_returns_true_none(self, connector):
        req = MockRequest("req-002")
        result = connector.request_finished(req)
        assert result == (True, None)


class TestGetNumNewMatchedTokens:
    def test_returns_stored_count(self, connector):
        from avp.connectors.vllm_kv_connector import compute_request_hash
        prompt_ids = [10, 20, 30]
        store_key = compute_request_hash(prompt_ids)
        connector._store.save_meta(store_key, seq_len=50, num_layers=2)

        req = MockRequest("req-match", prompt_token_ids=prompt_ids)
        matched, is_async = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
        # Capped at prompt_len - 1 to leave 1 token for scheduler
        assert matched == 2  # 3 tokens prompt, leave 1 → min(50, 3-0-1) = 2
        assert is_async is False

    def test_returns_zero_when_not_stored(self, connector):
        req = MockRequest("req-none", prompt_token_ids=[99, 98])
        matched, is_async = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
        assert matched == 0

    def test_subtracts_computed_tokens(self, connector):
        from avp.connectors.vllm_kv_connector import compute_request_hash
        # Use a longer prompt so the cap doesn't kick in
        prompt_ids = list(range(100))
        store_key = compute_request_hash(prompt_ids)
        connector._store.save_meta(store_key, seq_len=30, num_layers=1)

        req = MockRequest("req-partial", prompt_token_ids=prompt_ids)
        matched, _ = connector.get_num_new_matched_tokens(req, num_computed_tokens=10)
        assert matched == 20  # 30 - 10 = 20, capped at 100-10-1=89, so 20


class TestUpdateStateAfterAlloc:
    def test_records_pending_load(self, connector):
        req = MockRequest("req-alloc", prompt_token_ids=[1, 2])

        class MockBlocks:
            def get_block_ids(self):
                return ([5, 6, 7],)

        connector.update_state_after_alloc(req, MockBlocks(), num_external_tokens=10)
        from avp.connectors.vllm_kv_connector import _PENDING_LOADS
        assert "req-alloc" in _PENDING_LOADS
        assert _PENDING_LOADS["req-alloc"].block_ids == [5, 6, 7]
        assert _PENDING_LOADS["req-alloc"].num_external_tokens == 10

    def test_ignores_zero_external(self, connector):
        req = MockRequest("req-zero")
        connector.update_state_after_alloc(req, None, num_external_tokens=0)
        from avp.connectors.vllm_kv_connector import _PENDING_LOADS
        assert "req-zero" not in _PENDING_LOADS


class TestStartLoadKV:
    def test_injects_stored_kv(self, connector):
        """Save to store manually, then inject via start_load_kv."""
        block_size = connector._block_size  # 16
        num_blocks = 20
        num_kv_heads, head_dim = 2, 8
        features = num_kv_heads * head_dim

        # Manually save KV to store (simulating save_kv_layer + wait_for_save)
        store_key = "agent-a-key"
        kv_data_0 = torch.randn(2, 4, features)  # 4 tokens
        kv_data_1 = torch.randn(2, 4, features)
        connector._store.save_layer(store_key, 0, kv_data_0)
        connector._store.save_layer(store_key, 1, kv_data_1)
        connector._store.save_meta(store_key, seq_len=4, num_layers=2)

        # Set up Agent B's KV caches
        fresh_kv_0 = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_dim)
        fresh_kv_1 = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_dim)
        connector._kv_caches = {
            "model.layers.0.self_attn.attn": fresh_kv_0,
            "model.layers.1.self_attn.attn": fresh_kv_1,
        }

        # Schedule load (blocks at block 5)
        from avp.connectors.vllm_kv_connector import AVPReqMeta
        meta_b = AVPReqMeta(
            request_id="req-b", store_key=store_key,
            num_external_tokens=4, block_ids=[5],
        )
        from avp.connectors.vllm_kv_connector import _PENDING_LOADS
        _PENDING_LOADS["req-b"] = meta_b

        connector.start_load_kv(forward_context=MockForwardContext())

        # Verify injection at block 5 (slot 80 = 5 * 16)
        total_slots = num_blocks * block_size
        flat = fresh_kv_0.reshape(2, total_slots, -1)
        slot_80 = 5 * block_size
        assert flat[:, slot_80, :].abs().sum().item() > 0
        assert torch.allclose(flat[:, slot_80:slot_80 + 4, :], kv_data_0)


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

    def test_save_kv_layer_no_context(self, connector):
        """save_kv_layer returns early without ForwardContext (no vLLM runtime)."""
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
