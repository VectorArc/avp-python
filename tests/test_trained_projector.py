"""Tests for the trained cross-model projector (C2C).

Tests LayerProjector, TrainConfig, trained_hooks, and registry serialization
of trained AVPMap fields.
"""

import pytest
import torch
import torch.nn as nn

from avp.rosetta.train import LayerProjector, TrainConfig
from avp.rosetta.trained_hooks import trained_multi_layer_hook, _get_decoder_layers
from avp.rosetta.calibrate import AVPMap
from avp.types import ProjectionMethod


# --- TrainConfig tests ---


class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig()
        assert config.num_samples == 5000
        assert config.batch_size == 4
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 2
        assert config.gate_reg_weight == 0.01
        assert config.gate_init == -3.0
        assert config.max_seq_len == 256
        assert config.warmup_steps == 100
        assert config.seed == 42
        assert config.mse_aux_weight == 0.1
        assert config.use_ntp_loss is False

    def test_custom_values(self):
        config = TrainConfig(num_samples=100, batch_size=8, learning_rate=1e-3)
        assert config.num_samples == 100
        assert config.batch_size == 8
        assert config.learning_rate == 1e-3


# --- LayerProjector tests ---


class TestLayerProjector:
    def test_init_shapes(self):
        proj = LayerProjector(source_dim=128, target_dim=64, num_layers=4)
        assert len(proj.layer_projections) == 4
        assert len(proj.layer_gates) == 4
        # Each projection: [D_src] -> [D_tgt]
        assert proj.layer_projections[0].in_features == 128
        assert proj.layer_projections[0].out_features == 64

    def test_gate_init_near_zero(self):
        """Gates should be initialized near zero (sigmoid(-3) ~ 0.05)."""
        proj = LayerProjector(source_dim=32, target_dim=32, num_layers=8)
        for gate_logit in proj.layer_gates:
            gate_val = torch.sigmoid(gate_logit).item()
            assert gate_val < 0.1, f"Gate initialized too high: {gate_val}"

    def test_forward_output_shapes(self):
        proj = LayerProjector(source_dim=64, target_dim=32, num_layers=3)
        src = torch.randn(2, 64)  # batch=2
        results = proj.forward(src)
        assert len(results) == 3
        for projected, gate in results:
            assert projected.shape == (2, 32)
            assert 0 <= gate <= 1

    def test_get_active_layers_initial(self):
        """Initially, all gates near zero → no active layers at default threshold."""
        proj = LayerProjector(source_dim=32, target_dim=32, num_layers=8)
        active = proj.get_active_layers(threshold=0.1)
        assert len(active) == 0

    def test_get_active_layers_after_setting_gate(self):
        proj = LayerProjector(source_dim=32, target_dim=32, num_layers=4)
        # Force gate 1 and 3 open
        with torch.no_grad():
            proj.layer_gates[1].fill_(5.0)  # sigmoid(5) ~ 0.993
            proj.layer_gates[3].fill_(3.0)  # sigmoid(3) ~ 0.953
        active = proj.get_active_layers(threshold=0.5)
        assert active == [1, 3]

    def test_export_weights(self):
        proj = LayerProjector(source_dim=64, target_dim=32, num_layers=3)
        weights, biases, gates = proj.export_weights()
        assert len(weights) == 3
        assert len(biases) == 3
        assert len(gates) == 3
        for w in weights:
            assert w.shape == (32, 64)  # nn.Linear stores [out, in]
        for b in biases:
            assert b.shape == (32,)
        for g in gates:
            assert isinstance(g, float)
            assert 0 <= g <= 1

    def test_parameters_count(self):
        proj = LayerProjector(source_dim=64, target_dim=32, num_layers=3)
        params = list(proj.parameters())
        # 3 layers × (weight + bias) + 3 gates = 9 parameters
        assert len(params) == 9

    def test_to_device(self):
        proj = LayerProjector(source_dim=32, target_dim=16, num_layers=2)
        proj.to("cpu")  # should work without error
        # Verify parameters are on cpu
        for p in proj.parameters():
            assert str(p.device) == "cpu"


# --- Trained hooks tests ---


class FakeDecoderLayer(nn.Module):
    """Minimal decoder layer for hook testing."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return (self.linear(x),)


class FakeModel(nn.Module):
    """Minimal model with decoder layers for hook testing."""
    def __init__(self, dim, num_layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            FakeDecoderLayer(dim) for _ in range(num_layers)
        ])


class TestTrainedMultiLayerHook:
    def test_hook_adds_projection(self):
        dim = 16
        model = FakeModel(dim=dim, num_layers=4)
        model.eval()

        projected = torch.ones(1, dim) * 10.0
        gate = 0.5

        layer_projections = [None, (projected, gate), None, None]

        x = torch.randn(1, 5, dim)

        with trained_multi_layer_hook(model, layer_projections):
            # Forward through layer 1 (the hooked layer)
            layers = _get_decoder_layers(model)
            output = layers[1](x)

        # The hook should have modified the last token
        modified_hidden = output[0]
        # Verify last token was modified (added gate * projection)
        # Original output + 0.5 * 10.0 = original + 5.0
        with torch.no_grad():
            original_output = model.model.layers[1](x)
        original_last = original_output[0][:, -1, :]
        modified_last = modified_hidden[:, -1, :]
        diff = (modified_last - original_last).abs().mean().item()
        assert diff > 1.0, f"Hook didn't modify output enough: diff={diff}"

    def test_hook_fires_once(self):
        dim = 8
        model = FakeModel(dim=dim, num_layers=2)
        model.eval()

        projected = torch.ones(1, dim) * 100.0
        layer_projections = [(projected, 1.0), None]

        x = torch.randn(1, 3, dim)

        with trained_multi_layer_hook(model, layer_projections):
            layers = _get_decoder_layers(model)
            # First forward — hook fires
            out1 = layers[0](x)
            # Second forward — hook should NOT fire
            out2 = layers[0](x)

        # out1 should differ from out2 (hook only fires once)
        # Actually both outputs are from the same input x, but the first
        # has the injection and the second doesn't
        last1 = out1[0][:, -1, :].detach()
        last2 = out2[0][:, -1, :].detach()
        diff = (last1 - last2).abs().mean().item()
        assert diff > 1.0, f"Hook fired more than once or didn't fire: diff={diff}"

    def test_hook_cleanup(self):
        dim = 8
        model = FakeModel(dim=dim, num_layers=2)

        projected = torch.ones(1, dim)
        layer_projections = [(projected, 0.5), (projected, 0.3)]

        with trained_multi_layer_hook(model, layer_projections):
            pass

        # Verify hooks are removed
        layers = _get_decoder_layers(model)
        for layer in layers:
            assert len(layer._forward_hooks) == 0

    def test_skip_none_layers(self):
        dim = 8
        model = FakeModel(dim=dim, num_layers=4)

        # Only layer 2 active
        layer_projections = [None, None, (torch.ones(1, dim), 0.5), None]

        with trained_multi_layer_hook(model, layer_projections):
            layers = _get_decoder_layers(model)
            # Layers 0, 1, 3 should have no hooks
            assert len(layers[0]._forward_hooks) == 0
            assert len(layers[1]._forward_hooks) == 0
            assert len(layers[2]._forward_hooks) == 1
            assert len(layers[3]._forward_hooks) == 0

    def test_exception_cleanup(self):
        dim = 8
        model = FakeModel(dim=dim, num_layers=2)

        projected = torch.ones(1, dim)
        layer_projections = [(projected, 0.5), (projected, 0.3)]

        with pytest.raises(RuntimeError):
            with trained_multi_layer_hook(model, layer_projections):
                raise RuntimeError("test error")

        # Hooks should still be cleaned up
        layers = _get_decoder_layers(model)
        for layer in layers:
            assert len(layer._forward_hooks) == 0


# --- Registry serialization tests ---


class TestRegistrySerialization:
    def test_save_load_trained_map(self, tmp_path):
        """Round-trip save/load of AVPMap with trained projection fields."""
        from avp.rosetta.registry import save_map, load_map

        num_layers = 3
        src_dim = 64
        tgt_dim = 32

        avp_map = AVPMap(
            source_model_id="test/source",
            source_hash="abc123" * 8,
            source_dim=src_dim,
            target_model_id="test/target",
            target_hash="def456" * 8,
            target_dim=tgt_dim,
            w_map=torch.zeros(1),
            bias=None,
            target_norm=torch.tensor(1.0),
            method=ProjectionMethod.TRAINED,
            anchor_count=5000,
            validation_score=0.85,
            layer_weights=[torch.randn(tgt_dim, src_dim) for _ in range(num_layers)],
            layer_biases=[torch.randn(tgt_dim) for _ in range(num_layers)],
            layer_gates=[0.01, 0.95, 0.03],
        )

        path = save_map(avp_map, map_dir=tmp_path)
        assert path.exists()

        loaded = load_map(
            avp_map.source_hash, avp_map.target_hash,
            device="cpu", map_dir=tmp_path,
        )
        assert loaded is not None
        assert loaded.method == ProjectionMethod.TRAINED
        assert len(loaded.layer_weights) == num_layers
        assert len(loaded.layer_biases) == num_layers
        assert loaded.layer_gates == [0.01, 0.95, 0.03]

        # Verify tensor shapes preserved
        for w in loaded.layer_weights:
            assert w.shape == (tgt_dim, src_dim)
        for b in loaded.layer_biases:
            assert b.shape == (tgt_dim,)

    def test_save_load_non_trained_map_no_layer_fields(self, tmp_path):
        """Non-trained maps should have None for layer fields."""
        from avp.rosetta.registry import save_map, load_map

        avp_map = AVPMap(
            source_model_id="test/source",
            source_hash="aaa111" * 8,
            source_dim=64,
            target_model_id="test/target",
            target_hash="bbb222" * 8,
            target_dim=32,
            w_map=torch.randn(64, 32),
            bias=None,
            target_norm=torch.tensor(1.0),
            method=ProjectionMethod.RIDGE,
            anchor_count=50,
            validation_score=0.5,
        )

        save_map(avp_map, map_dir=tmp_path)
        loaded = load_map(
            avp_map.source_hash, avp_map.target_hash,
            device="cpu", map_dir=tmp_path,
        )
        assert loaded is not None
        assert loaded.layer_weights is None
        assert loaded.layer_biases is None
        assert loaded.layer_gates is None


# --- AVPMap TRAINED method test ---


class TestAVPMapTrained:
    def test_trained_method_enum(self):
        assert ProjectionMethod.TRAINED.value == "trained"

    def test_avpmap_with_trained_fields(self):
        avp_map = AVPMap(
            source_model_id="src",
            source_hash="h1",
            source_dim=128,
            target_model_id="tgt",
            target_hash="h2",
            target_dim=64,
            w_map=torch.zeros(1),
            bias=None,
            target_norm=torch.tensor(1.0),
            method="trained",  # string should convert
            anchor_count=1000,
            validation_score=0.9,
            layer_weights=[torch.randn(64, 128)],
            layer_biases=[torch.randn(64)],
            layer_gates=[0.5],
        )
        assert avp_map.method == ProjectionMethod.TRAINED
