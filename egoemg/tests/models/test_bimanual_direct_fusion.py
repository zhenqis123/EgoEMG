"""Matched direct fusion: no angle addition, real gradients, missing EMG."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

from egoemg.models.modules.bimanual_direct_fusion import BimanualDirectFusion
from egoemg.models.modules.bimanual_simple_residual import BimanualSimpleResidualFusion
from egoemg.tests.models.test_bimanual_fusion import Emg, Vision
from egoemg.tests.models.test_bimanual_simple_residual import batch


def make_pair(monkeypatch):
    def load(path):
        model = Vision() if path == "vision" else Emg()
        if path != "vision":
            model.head = nn.Linear(256, 22)
        return model

    monkeypatch.setattr("egoemg.models.modules.bimanual_fusion.load_branch", load)
    torch.manual_seed(42)
    residual = BimanualSimpleResidualFusion(
        "vision", "emg", freeze_vision=False, independent_hand_heads=True
    )
    rng_residual = torch.random.get_rng_state()
    torch.manual_seed(42)
    direct = BimanualDirectFusion(
        "vision", "emg", freeze_vision=False, independent_hand_heads=True
    )
    assert torch.equal(rng_residual, torch.random.get_rng_state())
    return residual, direct


def test_matched_features_and_direct_prediction_without_angle_addition(monkeypatch):
    residual, direct = make_pair(monkeypatch)
    for key, value in residual.state_dict().items():
        if key.startswith("head.") and ".net.3." in key:
            continue
        torch.testing.assert_close(value, direct.state_dict()[key], rtol=0, atol=0)
    residual.load_state_dict(direct.state_dict())
    residual.eval()
    direct.eval()
    data = batch()
    prediction = direct(data)[0]
    baseline = direct._last_vision_prediction.clone()
    torch.testing.assert_close(residual(data)[0], prediction + baseline)
    with torch.no_grad():
        direct.vision.head[0].bias.add_(100)
    torch.testing.assert_close(direct(data)[0], prediction, rtol=0, atol=0)


def test_direct_gradients_and_missing_emg_zero_features(monkeypatch):
    _, direct = make_pair(monkeypatch)
    data = batch()
    prediction, target, valid = direct(data)
    assert prediction.shape == target.shape == valid.shape == (4, 22, 1)
    (prediction - target).abs().mean().backward()
    for layer in [direct.vision.backbone, direct.emg_model.featurizer,
                  direct.emg_model.decoder, direct.temporal_attn,
                  direct.vision_proj, direct.fusion_proj, direct.head]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in layer.parameters())
    assert all(p.grad is None for p in direct.vision.head.parameters())
    assert all(p.grad is None for p in direct.emg_model.head.parameters())
    direct.eval()
    data["emg_valid_mask"].zero_()
    data["emg"].fill_(float("nan"))
    prediction = direct(data)[0]
    assert torch.isfinite(prediction).all()
    assert not torch.allclose(prediction, direct._last_vision_prediction)
    data["emg"].zero_()
    torch.testing.assert_close(direct(data)[0], prediction, rtol=0, atol=0)


def test_ablation_configs_match_except_prediction_mode():
    config_dir = str(Path(__file__).resolve().parents[3] / "config")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        configs = [compose(config_name="base", overrides=[
            "experiment=fusion/" + {
                "residual": "fusion_vit_s",
                "direct": "direct_fusion_vit_s",
            }[mode]
        ]) for mode in ["residual", "direct"]]
    a, b = configs
    for key in ["dataset", "datamodule", "optimizer", "lr_scheduler",
                "component_lr_scales", "trainer", "callbacks", "seed",
                "batch_size", "transforms", "loss_weights"]:
        assert a[key] == b[key]
    for key in a.module:
        if key != "_target_":
            assert a.module[key] == b.module[key]
    assert a.datamodule.train_samples_per_epoch == 168928
    assert a.module.independent_hand_heads
    assert a.optimizer.lr == 1e-4


def test_common_valid_emg_metrics_exclude_missing_windows_and_invalid_angles():
    from egoemg.training.vision import ResidualFusionPredictionModule

    module = ResidualFusionPredictionModule.__new__(ResidualFusionPredictionModule)
    nn.Module.__init__(module)
    module.register_buffer('_gate_eval_sums', torch.zeros(5, 3, dtype=torch.float64))
    module.register_buffer('_gate_valid_emg_sums', torch.zeros(3, dtype=torch.float64))
    prediction = torch.tensor([[[1., 2.], [3., 4.]]])
    valid = torch.tensor([[[True, False], [True, True]]])
    module._gate_step = lambda data: (
        torch.tensor(0.), prediction, torch.zeros_like(prediction), valid,
        {'vision_prediction': prediction + 1},
    )
    logged = {}
    module.log = lambda name, value, **kwargs: logged.update({name: value.item()})
    module.validation_step(dict(split_id=torch.tensor([[1, 2]]),
                                emg_valid_mask=torch.tensor([[True, False]])), 0)
    module._finish_gate_metrics('val')
    assert logged['val_mae'] == 8/3
    assert logged['val_valid_emg_mae'] == 1
    assert logged['val_valid_emg_vision_mae'] == 2
    assert logged['val_valid_emg_fraction'] == 1/3
