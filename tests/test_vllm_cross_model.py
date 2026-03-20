"""Tests for cross-model rosetta projection via vLLM (mock-based, no GPU required).

Tests the pipeline: think → rosetta projection → save → load.
"""

import os
import tempfile

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
# Mock infrastructure
# ---------------------------------------------------------------------------


class MockConfig:
    """Mock model config."""

    def __init__(self, hidden_size=64, vocab_size=100, tie_word_embeddings=True):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self._name_or_path = "mock/source-model"

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "_name_or_path": self._name_or_path,
        }


class MockTokenizer:
    """Mock tokenizer with a controllable vocabulary."""

    def __init__(self, vocab: dict):
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)


class MockEmbedding:
    def __init__(self, vocab_size, hidden_size):
        self.weight = torch.randn(vocab_size, hidden_size)


class MockModel:
    def __init__(self, hidden_size=64, vocab_size=100):
        self.embed_tokens = MockEmbedding(vocab_size, hidden_size)


class MockLMHead:
    def __init__(self, vocab_size, hidden_size):
        self.weight = torch.randn(vocab_size, hidden_size)


# ---------------------------------------------------------------------------
# Tests: calibrate_from_weights
# ---------------------------------------------------------------------------


class TestCalibrateFromWeights:
    """Test AVPMap creation from raw weight tensors."""

    def test_vocab_mediated_same_tokenizer(self):
        """Same tokenizer → vocab-mediated projection."""
        from avp.rosetta.calibrate import calibrate_from_weights
        from avp.types import ProjectionMethod

        vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
        src_tokenizer = MockTokenizer(vocab)
        tgt_tokenizer = MockTokenizer(vocab)  # same vocab

        tgt_embed = torch.randn(4, 32)
        src_config = {"hidden_size": 64, "vocab_size": 4}
        tgt_config = {"hidden_size": 32, "vocab_size": 4}

        avp_map = calibrate_from_weights(
            source_model_id="mock/src",
            source_config_dict=src_config,
            target_model_id="mock/tgt",
            target_config_dict=tgt_config,
            target_embed_weight=tgt_embed,
            source_tokenizer=src_tokenizer,
            target_tokenizer=tgt_tokenizer,
            auto_save=False,
        )

        assert avp_map.method == ProjectionMethod.VOCAB_MEDIATED
        assert avp_map.source_dim == 64
        assert avp_map.target_dim == 32
        assert avp_map.w_map.shape == (4, 32)
        assert avp_map.validation_score == 1.0

    def test_vocab_overlap_different_tokenizers(self):
        """Different tokenizers with overlap → vocab-overlap projection."""
        from avp.rosetta.calibrate import calibrate_from_weights
        from avp.types import ProjectionMethod

        # Build vocabularies with >100 shared tokens
        shared = {f"tok_{i}": i for i in range(150)}
        src_only = {f"src_{i}": 150 + i for i in range(50)}
        tgt_only = {f"tgt_{i}": 150 + i for i in range(50)}

        src_vocab = {**shared, **src_only}
        tgt_vocab = {**shared, **tgt_only}

        src_tokenizer = MockTokenizer(src_vocab)
        tgt_tokenizer = MockTokenizer(tgt_vocab)

        tgt_embed = torch.randn(200, 32)
        src_config = {"hidden_size": 64, "vocab_size": 200}
        tgt_config = {"hidden_size": 32, "vocab_size": 200}

        avp_map = calibrate_from_weights(
            source_model_id="mock/src",
            source_config_dict=src_config,
            target_model_id="mock/tgt",
            target_config_dict=tgt_config,
            target_embed_weight=tgt_embed,
            source_tokenizer=src_tokenizer,
            target_tokenizer=tgt_tokenizer,
            auto_save=False,
        )

        assert avp_map.method == ProjectionMethod.VOCAB_OVERLAP
        assert avp_map.overlap_count == 150
        assert avp_map.src_indices is not None
        assert avp_map.tgt_indices is not None
        assert avp_map.w_map.shape[0] == 150  # N_shared
        assert avp_map.w_map.shape[1] == 32  # D_tgt

    def test_insufficient_overlap_raises(self):
        """Too few shared tokens → ValueError."""
        from avp.rosetta.calibrate import calibrate_from_weights

        src_vocab = {f"src_{i}": i for i in range(100)}
        tgt_vocab = {f"tgt_{i}": i for i in range(100)}  # no overlap

        src_tokenizer = MockTokenizer(src_vocab)
        tgt_tokenizer = MockTokenizer(tgt_vocab)

        with pytest.raises(ValueError, match="Insufficient vocabulary overlap"):
            calibrate_from_weights(
                source_model_id="mock/src",
                source_config_dict={"hidden_size": 64, "vocab_size": 100},
                target_model_id="mock/tgt",
                target_config_dict={"hidden_size": 32, "vocab_size": 100},
                target_embed_weight=torch.randn(100, 32),
                source_tokenizer=src_tokenizer,
                target_tokenizer=tgt_tokenizer,
                auto_save=False,
            )

    def test_target_norm_computed_from_weights(self):
        """target_norm should be the mean L2 norm of embed weights."""
        from avp.rosetta.calibrate import calibrate_from_weights

        vocab = {"a": 0, "b": 1, "c": 2}
        tokenizer = MockTokenizer(vocab)
        tgt_embed = torch.randn(3, 16)
        expected_norm = tgt_embed.float().norm(dim=1).mean()

        avp_map = calibrate_from_weights(
            source_model_id="mock/src",
            source_config_dict={"hidden_size": 32, "vocab_size": 3},
            target_model_id="mock/tgt",
            target_config_dict={"hidden_size": 16, "vocab_size": 3},
            target_embed_weight=tgt_embed,
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
            auto_save=False,
        )

        assert torch.allclose(avp_map.target_norm, expected_norm, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: FileKVStore projected embedding
# ---------------------------------------------------------------------------


class TestFileKVStoreProjected:
    """Test projected embedding save/load in FileKVStore."""

    def test_save_and_load(self, tmp_path):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(str(tmp_path))
        emb = torch.randn(32)  # [D_tgt]
        store.save_projected("test-key", emb)

        loaded = store.load_projected("test-key")
        assert loaded is not None
        assert torch.allclose(loaded, emb)

    def test_load_missing_returns_none(self, tmp_path):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(str(tmp_path))
        assert store.load_projected("nonexistent") is None

    def test_has_projected(self, tmp_path):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(str(tmp_path))
        assert store.has_projected("key") is False
        store.save_projected("key", torch.randn(16))
        assert store.has_projected("key") is True

    def test_save_2d_tensor(self, tmp_path):
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(str(tmp_path))
        emb = torch.randn(1, 32)  # [1, D_tgt]
        store.save_projected("key2d", emb)
        loaded = store.load_projected("key2d")
        assert loaded is not None
        assert torch.allclose(loaded, emb)


# ---------------------------------------------------------------------------
# Tests: load_projected_embedding helper
# ---------------------------------------------------------------------------


class TestLoadProjectedEmbedding:
    def test_load_from_store(self, tmp_path):
        from avp.connectors.vllm_kv_connector import (
            FileKVStore,
            load_projected_embedding,
        )

        store = FileKVStore(str(tmp_path))
        emb = torch.randn(48)
        store.save_projected("my-key", emb)

        loaded = load_projected_embedding(str(tmp_path), "my-key")
        assert loaded is not None
        assert torch.allclose(loaded, emb)

    def test_load_missing_returns_none(self, tmp_path):
        from avp.connectors.vllm_kv_connector import load_projected_embedding

        assert load_projected_embedding(str(tmp_path), "missing") is None


# ---------------------------------------------------------------------------
# Tests: Stub cross-model projection
# ---------------------------------------------------------------------------


class TestStubCrossModelProjection:
    """Test the stub's cross-model projection method."""

    def _make_cross_model_stub(
        self, src_hidden=64, src_vocab=200, tgt_hidden=32, tgt_vocab=200,
    ):
        """Create a stub configured for cross-model projection."""
        from avp.rosetta.calibrate import calibrate_from_weights
        from avp.connectors.vllm_model_plugin import _AVPLatentStub

        os.environ["AVP_LATENT_STEPS"] = "3"
        os.environ.pop("AVP_TARGET_MODEL", None)

        # Build shared vocab with >100 tokens
        shared = {f"tok_{i}": i for i in range(150)}
        src_only = {f"src_{i}": 150 + i for i in range(src_vocab - 150)}
        tgt_only = {f"tgt_{i}": 150 + i for i in range(tgt_vocab - 150)}

        src_tokenizer = MockTokenizer({**shared, **src_only})
        tgt_tokenizer = MockTokenizer({**shared, **tgt_only})

        tgt_embed = torch.randn(tgt_vocab, tgt_hidden)
        src_lm_head = torch.randn(src_vocab, src_hidden)

        avp_map = calibrate_from_weights(
            source_model_id="mock/src",
            source_config_dict={"hidden_size": src_hidden, "vocab_size": src_vocab},
            target_model_id="mock/tgt",
            target_config_dict={"hidden_size": tgt_hidden, "vocab_size": tgt_vocab},
            target_embed_weight=tgt_embed,
            source_tokenizer=src_tokenizer,
            target_tokenizer=tgt_tokenizer,
            auto_save=False,
        )

        config = MockConfig(hidden_size=src_hidden, vocab_size=src_vocab)
        model = MockModel(hidden_size=src_hidden, vocab_size=src_vocab)
        lm_head = MockLMHead(src_vocab, src_hidden)

        stub = _AVPLatentStub(config=config, model=model, lm_head=lm_head)
        stub._avp_map = avp_map
        stub._source_lm_head_cpu = src_lm_head.cpu().float()
        stub._cross_model_ready = True

        return stub

    def test_projection_produces_target_dim(self):
        """Projected embedding should have target model's dimension."""
        stub = self._make_cross_model_stub(src_hidden=64, tgt_hidden=32)
        hidden = torch.randn(1, 64)

        projected = stub._project_cross_model_stub(hidden)
        assert projected is not None
        assert projected.shape[-1] == 32

    def test_projection_with_1d_input(self):
        """1D hidden state should also work."""
        stub = self._make_cross_model_stub(src_hidden=64, tgt_hidden=32)
        hidden = torch.randn(64)

        projected = stub._project_cross_model_stub(hidden)
        assert projected is not None
        assert projected.shape[-1] == 32

    def test_projection_none_without_avp_map(self):
        """Returns None when AVPMap is not set."""
        os.environ["AVP_LATENT_STEPS"] = "3"
        os.environ.pop("AVP_TARGET_MODEL", None)

        from avp.connectors.vllm_model_plugin import _AVPLatentStub

        stub = _AVPLatentStub(
            config=MockConfig(),
            model=MockModel(),
        )

        result = stub._project_cross_model_stub(torch.randn(1, 64))
        assert result is None

    def test_projected_is_normalized(self):
        """Projected embedding should be normalized to target_norm."""
        stub = self._make_cross_model_stub(src_hidden=64, tgt_hidden=32)
        hidden = torch.randn(1, 64)

        projected = stub._project_cross_model_stub(hidden)
        assert projected is not None

        proj_norm = projected.float().norm(dim=-1)
        target_norm = stub._avp_map.target_norm.float()
        assert torch.allclose(proj_norm, target_norm, atol=0.1)


# ---------------------------------------------------------------------------
# Tests: Connector flushes projected embeddings
# ---------------------------------------------------------------------------


class TestConnectorProjectedFlush:
    """Test that the connector flushes projected embeddings from the model plugin."""

    def test_flush_projected_in_wait_for_save(self):
        """wait_for_save should flush _PROJECTED_EMBEDDINGS to store."""
        from avp.connectors.vllm_kv_connector import (
            AVPKVConnectorV1Dynamic,
            FileKVStore,
            _PROJECTED_EMBEDDINGS,
        )

        with tempfile.TemporaryDirectory() as store_dir:
            os.environ["AVP_KV_STORE_DIR"] = store_dir
            os.environ.pop("AVP_LATENT_STEPS", None)
            os.environ.pop("AVP_TARGET_MODEL", None)

            class MockKVConfig:
                kv_connector_extra_config = {
                    "avp_store_dir": store_dir,
                    "avp_latent_steps": 0,
                }

            class MockVLLMConfig:
                kv_transfer_config = MockKVConfig()

            connector = AVPKVConnectorV1Dynamic(
                vllm_config=MockVLLMConfig(), role=None,
            )

            # Simulate model plugin writing a projected embedding
            emb = torch.randn(32)
            _PROJECTED_EMBEDDINGS["test-flush-key"] = emb

            connector.wait_for_save()

            # Wait for background flush thread
            if connector._save_thread is not None:
                connector._save_thread.join(timeout=5.0)

            # Verify the projected embedding was saved to the store
            store = FileKVStore(store_dir)
            loaded = store.load_projected("test-flush-key")
            assert loaded is not None
            assert torch.allclose(loaded, emb)

            # The module-level dict should be cleared
            assert "test-flush-key" not in _PROJECTED_EMBEDDINGS


# ---------------------------------------------------------------------------
# Tests: End-to-end pipeline (think → project → save → load)
# ---------------------------------------------------------------------------


class TestEndToEndCrossModel:
    """Test the full cross-model pipeline with mocks."""

    def test_think_project_save_load(self):
        """Full pipeline: stub thinks, projects, connector saves, helper loads."""
        from avp.connectors.vllm_kv_connector import (
            AVPKVConnectorV1Dynamic,
            _PROJECTED_EMBEDDINGS,
            compute_request_hash,
            load_projected_embedding,
        )

        with tempfile.TemporaryDirectory() as store_dir:
            os.environ.pop("AVP_TARGET_MODEL", None)

            class MockKVConfig:
                kv_connector_extra_config = {
                    "avp_store_dir": store_dir,
                    "avp_latent_steps": 0,
                }

            class MockVLLMConfig:
                kv_transfer_config = MockKVConfig()

            connector = AVPKVConnectorV1Dynamic(
                vllm_config=MockVLLMConfig(), role=None,
            )

            # Step 1: Create cross-model stub and project
            from avp.connectors.vllm_model_plugin import _AVPLatentStub
            from avp.rosetta.calibrate import calibrate_from_weights

            src_hidden, tgt_hidden = 64, 32
            src_vocab, tgt_vocab = 200, 200

            shared = {f"tok_{i}": i for i in range(150)}
            src_only = {f"src_{i}": 150 + i for i in range(50)}
            tgt_only = {f"tgt_{i}": 150 + i for i in range(50)}

            avp_map = calibrate_from_weights(
                source_model_id="mock/src",
                source_config_dict={"hidden_size": src_hidden, "vocab_size": src_vocab},
                target_model_id="mock/tgt",
                target_config_dict={"hidden_size": tgt_hidden, "vocab_size": tgt_vocab},
                target_embed_weight=torch.randn(tgt_vocab, tgt_hidden),
                source_tokenizer=MockTokenizer({**shared, **src_only}),
                target_tokenizer=MockTokenizer({**shared, **tgt_only}),
                auto_save=False,
            )

            os.environ["AVP_LATENT_STEPS"] = "2"
            stub = _AVPLatentStub(
                config=MockConfig(hidden_size=src_hidden, vocab_size=src_vocab),
                model=MockModel(hidden_size=src_hidden, vocab_size=src_vocab),
                lm_head=MockLMHead(src_vocab, src_hidden),
            )
            stub._avp_map = avp_map
            stub._source_lm_head_cpu = torch.randn(src_vocab, src_hidden).float()
            stub._cross_model_ready = True

            # Step 2: Think (generate enriched hidden state) and project
            hidden = torch.randn(1, src_hidden)
            projected = stub._project_cross_model_stub(hidden)
            assert projected is not None

            # Step 3: Simulate the model plugin storing the projected embedding
            prompt_ids = [1, 2, 3, 4, 5]
            store_key = compute_request_hash(prompt_ids)
            _PROJECTED_EMBEDDINGS[store_key] = projected.squeeze(0)

            # Step 4: Connector flushes to store
            connector.wait_for_save()
            if connector._save_thread is not None:
                connector._save_thread.join(timeout=5.0)

            # Step 5: Agent B loads the projected embedding
            loaded = load_projected_embedding(store_dir, store_key)
            assert loaded is not None
            assert loaded.shape[-1] == tgt_hidden
            assert torch.allclose(loaded, projected.squeeze(0))

    def test_vocab_mediated_same_tokenizer_pipeline(self):
        """Same tokenizer uses vocab-mediated projection (zero calibration)."""
        from avp.connectors.vllm_model_plugin import _AVPLatentStub
        from avp.rosetta.calibrate import calibrate_from_weights
        from avp.types import ProjectionMethod

        src_hidden, tgt_hidden = 64, 32
        vocab_size = 100
        vocab = {f"tok_{i}": i for i in range(vocab_size)}
        tokenizer = MockTokenizer(vocab)

        tgt_embed = torch.randn(vocab_size, tgt_hidden)
        avp_map = calibrate_from_weights(
            source_model_id="mock/src",
            source_config_dict={"hidden_size": src_hidden, "vocab_size": vocab_size},
            target_model_id="mock/tgt",
            target_config_dict={"hidden_size": tgt_hidden, "vocab_size": vocab_size},
            target_embed_weight=tgt_embed,
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
            auto_save=False,
        )

        assert avp_map.method == ProjectionMethod.VOCAB_MEDIATED

        os.environ["AVP_LATENT_STEPS"] = "2"
        os.environ.pop("AVP_TARGET_MODEL", None)
        stub = _AVPLatentStub(
            config=MockConfig(hidden_size=src_hidden, vocab_size=vocab_size),
            model=MockModel(hidden_size=src_hidden, vocab_size=vocab_size),
            lm_head=MockLMHead(vocab_size, src_hidden),
        )
        stub._avp_map = avp_map
        stub._source_lm_head_cpu = torch.randn(vocab_size, src_hidden).float()
        stub._cross_model_ready = True

        hidden = torch.randn(1, src_hidden)
        projected = stub._project_cross_model_stub(hidden)
        assert projected is not None
        assert projected.shape[-1] == tgt_hidden


# ---------------------------------------------------------------------------
# Tests: Cross-model KV skip
# ---------------------------------------------------------------------------


class TestCrossModelKVSkip:
    """Test that save_kv_layer is skipped in cross-model mode."""

    def test_cross_model_skips_kv_save(self):
        """Connector with avp_target_model should skip save_kv_layer."""
        from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic

        with tempfile.TemporaryDirectory() as store_dir:
            class MockKVConfig:
                kv_connector_extra_config = {
                    "avp_store_dir": store_dir,
                    "avp_latent_steps": 10,
                    "avp_target_model": "some/target-model",
                }

            class MockVLLMConfig:
                kv_transfer_config = MockKVConfig()

            connector = AVPKVConnectorV1Dynamic(
                vllm_config=MockVLLMConfig(), role=None,
            )

            assert connector._cross_model_only is True

            # save_kv_layer should be a no-op (returns immediately)
            # Calling with garbage args — if it didn't return early, it would crash
            connector.save_kv_layer("model.layers.0.attn", None, None)

            # No pending saves should exist
            assert len(connector._pending_saves) == 0

    def test_same_model_does_not_skip_kv_save(self):
        """Connector without avp_target_model should NOT skip save_kv_layer."""
        from avp.connectors.vllm_kv_connector import AVPKVConnectorV1Dynamic

        with tempfile.TemporaryDirectory() as store_dir:
            os.environ.pop("AVP_TARGET_MODEL", None)

            class MockKVConfig:
                kv_connector_extra_config = {
                    "avp_store_dir": store_dir,
                    "avp_latent_steps": 10,
                }

            class MockVLLMConfig:
                kv_transfer_config = MockKVConfig()

            connector = AVPKVConnectorV1Dynamic(
                vllm_config=MockVLLMConfig(), role=None,
            )

            assert connector._cross_model_only is False


# ---------------------------------------------------------------------------
# Tests: Projected embedding synchronous flush
# ---------------------------------------------------------------------------


class TestProjectedSyncFlush:
    """Test that projected embeddings are flushed synchronously."""

    def test_projected_available_immediately_after_wait_for_save(self):
        """projected.pt should exist on disk immediately after wait_for_save."""
        from avp.connectors.vllm_kv_connector import (
            AVPKVConnectorV1Dynamic,
            FileKVStore,
            _PROJECTED_EMBEDDINGS,
        )

        with tempfile.TemporaryDirectory() as store_dir:
            os.environ.pop("AVP_TARGET_MODEL", None)

            class MockKVConfig:
                kv_connector_extra_config = {
                    "avp_store_dir": store_dir,
                    "avp_latent_steps": 0,
                }

            class MockVLLMConfig:
                kv_transfer_config = MockKVConfig()

            connector = AVPKVConnectorV1Dynamic(
                vllm_config=MockVLLMConfig(), role=None,
            )

            emb = torch.randn(64)
            _PROJECTED_EMBEDDINGS["sync-key"] = emb

            connector.wait_for_save()

            # No sleep needed — projected flush is synchronous
            store = FileKVStore(store_dir)
            loaded = store.load_projected("sync-key")
            assert loaded is not None
            assert torch.allclose(loaded, emb)


# ---------------------------------------------------------------------------
# Tests: Production hardening (audit fixes)
# ---------------------------------------------------------------------------


class TestAuditFixes:
    """Tests for issues found during the engineering audit."""

    def test_request_finished_cleans_store_keys(self):
        """C1: _REQUEST_STORE_KEYS entries are cleaned up in request_finished."""
        from avp.connectors.vllm_kv_connector import (
            AVPKVConnectorV1Dynamic,
            _REQUEST_STORE_KEYS,
        )

        with tempfile.TemporaryDirectory() as store_dir:
            os.environ.pop("AVP_TARGET_MODEL", None)

            class MockKVConfig:
                kv_connector_extra_config = {
                    "avp_store_dir": store_dir,
                    "avp_latent_steps": 0,
                }

            class MockVLLMConfig:
                kv_transfer_config = MockKVConfig()

            connector = AVPKVConnectorV1Dynamic(
                vllm_config=MockVLLMConfig(), role=None,
            )

            # Simulate build_connector_meta adding an entry
            _REQUEST_STORE_KEYS["req-cleanup-test"] = "some-store-key"
            assert "req-cleanup-test" in _REQUEST_STORE_KEYS

            # request_finished should clean it up
            class Req:
                request_id = "req-cleanup-test"

            connector.request_finished(Req())
            assert "req-cleanup-test" not in _REQUEST_STORE_KEYS

    def test_atomic_write_no_tmp_leftover(self, tmp_path):
        """C3: No .tmp files left after save completes."""
        from avp.connectors.vllm_kv_connector import FileKVStore

        store = FileKVStore(str(tmp_path))
        store.save_layer("atomic-test", 0, torch.randn(2, 4, 8))
        store.save_meta("atomic-test", seq_len=10, num_layers=1)
        store.save_projected("atomic-test", torch.randn(32))

        # No .tmp files should remain
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == [], f"Leftover tmp files: {tmp_files}"

    def test_temperature_zero_raises(self):
        """I4: temperature=0 should raise ValueError, not produce NaN."""
        from avp.rosetta.project import (
            vocab_overlap_projection,
            vocabulary_mediated_projection,
        )

        h = torch.randn(1, 64)
        w = torch.randn(100, 64)

        with pytest.raises(ValueError, match="temperature must be positive"):
            vocabulary_mediated_projection(h, w, w, temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            vocab_overlap_projection(
                h, w, w[:50], torch.arange(50), temperature=0.0,
            )

    def test_nan_embed_weight_raises(self):
        """I5: NaN embed weights should raise ValueError."""
        from avp.rosetta.calibrate import calibrate_from_weights

        vocab = {f"tok_{i}": i for i in range(150)}
        tokenizer = MockTokenizer(vocab)

        nan_embed = torch.full((150, 32), float("nan"))

        with pytest.raises(ValueError, match="Invalid target_norm"):
            calibrate_from_weights(
                source_model_id="mock/src",
                source_config_dict={"hidden_size": 64, "vocab_size": 150},
                target_model_id="mock/tgt",
                target_config_dict={"hidden_size": 32, "vocab_size": 150},
                target_embed_weight=nan_embed,
                source_tokenizer=tokenizer,
                target_tokenizer=tokenizer,
                auto_save=False,
            )

    def test_zero_hidden_size_raises(self):
        """I6: Missing hidden_size in config should raise ValueError."""
        from avp.rosetta.calibrate import calibrate_from_weights

        vocab = {f"tok_{i}": i for i in range(150)}
        tokenizer = MockTokenizer(vocab)

        with pytest.raises(ValueError, match="Invalid hidden dimensions"):
            calibrate_from_weights(
                source_model_id="mock/src",
                source_config_dict={"vocab_size": 150},  # no hidden_size
                target_model_id="mock/tgt",
                target_config_dict={"vocab_size": 150},  # no hidden_size
                target_embed_weight=torch.randn(150, 32),
                source_tokenizer=tokenizer,
                target_tokenizer=tokenizer,
                auto_save=False,
            )
