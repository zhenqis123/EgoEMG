"""Current data recipe with July-style learned temporal-pool residuals."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from egoemg.models.modules.bimanual_simple_residual import BimanualSimpleResidualFusion
from egoemg.tests.models.test_bimanual_fusion import Emg, Vision


@pytest.fixture
def experts(monkeypatch):
    vision, emg = Vision(), Emg()
    emg.head = nn.Conv1d(256, 22, 1)
    monkeypatch.setattr(
        "egoemg.models.modules.bimanual_fusion.load_branch",
        lambda path: vision if path == "vision" else emg,
    )
    return vision, emg


def batch():
    return dict(
        vision_img=torch.rand(2, 3, 16, 16),
        emg=torch.randn(2, 2, 8, 20),
        emg_valid_mask=torch.ones(2, 2, dtype=torch.bool),
        joint_angles=torch.randn(2, 44, 1),
        label_valid_mask=torch.ones(2, 44, 1, dtype=torch.bool),
    )


def baseline(model, data):
    return model.vision.head(model.vision.backbone(data["vision_img"])).reshape(
        -1, 22, 1
    )


def test_scratch_emg_never_loads_emg_checkpoint(monkeypatch):
    from egoemg.tests.models.test_bimanual_fusion import Emg

    vision = Vision()
    loaded = []

    def load(path):
        loaded.append(path)
        assert path == "vision"
        return vision

    monkeypatch.setattr("egoemg.models.modules.bimanual_fusion.load_branch", load)
    torch.manual_seed(42)
    model = BimanualSimpleResidualFusion(
        "vision", None, freeze_vision=False, freeze_emg=False,
        emg_config={"_target_": "egoemg.tests.models.test_bimanual_fusion.Emg"},
    )
    assert loaded == ["vision"]
    assert isinstance(model.emg_model, Emg)
    assert all(p.requires_grad for p in model.vision.parameters())
    assert all(p.requires_grad for p in model.emg_model.parameters())
    pred, target, _ = model(batch())
    (pred - target).abs().mean().backward()
    for layer in (model.vision.backbone, model.vision.head,
                  model.emg_model.featurizer, model.emg_model.decoder,
                  model.temporal_attn, model.head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in layer.parameters())
    with pytest.raises(ValueError, match="requires emg_config"):
        BimanualSimpleResidualFusion("vision", None)
    with pytest.raises(ValueError, match="not both"):
        BimanualSimpleResidualFusion("vision", "emg", emg_config={})


def test_near_vision_init_missing_nan_emg_and_zero_correction(experts):
    model = BimanualSimpleResidualFusion("vision", "emg").eval()
    data = batch()
    expected = baseline(model, data)
    pred, target, mask = model(data)
    assert pred.shape == target.shape == mask.shape == (4, 22, 1)
    assert (pred - expected).abs().max() < 1e-3
    assert model._last_delta.abs().sum() > 0
    data["emg_valid_mask"][0, 1] = False
    data["emg"][0, 1] = float("nan")
    with torch.no_grad():
        model.head.net[-1].bias.fill_(0.5)
    pred = model(data)[0]
    assert torch.isfinite(pred).all()
    torch.testing.assert_close(pred[1], expected[1], rtol=0, atol=0)
    model.force_zero_correction = True
    torch.testing.assert_close(model(data)[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("freeze_emg", [False, True])
def test_pool_and_fusion_gradients_with_trainable_or_frozen_emg(experts, freeze_emg):
    model = BimanualSimpleResidualFusion("vision", "emg", freeze_emg=freeze_emg).train()
    data = batch()
    before = {k: v.clone() for k, v in model.vision.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5
    )
    pred, target, _ = model(data)
    (pred - target).square().mean().backward()
    for layer in [
        model.temporal_attn,
        model.vision_proj,
        model.fusion_proj,
        model.head,
    ]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()
        )
    for layer in [model.emg_model.featurizer, model.emg_model.decoder]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()
        ) == (not freeze_emg)
    assert all(p.grad is None for p in model.vision.parameters())
    assert all(p.grad is None for p in model.emg_model.head.parameters())
    assert model.temporal_attn.training and not model.vision.training
    assert model.emg_model.training == (not freeze_emg)
    optimizer.step()
    for key, value in before.items():
        torch.testing.assert_close(
            model.vision.state_dict()[key], value, rtol=0, atol=0
        )


def test_hand_id_distinguishes_equal_signal_and_missing_windows_skip_bn(experts):
    model = BimanualSimpleResidualFusion("vision", "emg").eval()
    data = batch()
    data["emg"][:, 1] = data["emg"][:, 0]
    with torch.no_grad():
        model.head.net[-1].weight.normal_(std=0.02)
    model(data)
    assert not torch.equal(model._last_delta[0], model._last_delta[1])
    model.train()
    data["emg_valid_mask"].zero_()
    data["emg"].fill_(float("nan"))
    before = model.emg_model.featurizer[1].running_mean.clone()
    pred, target, _ = model(data)
    torch.testing.assert_close(pred, baseline(model, data), rtol=0, atol=0)
    (pred - target).abs().mean().backward()
    assert model.head.net[-1].weight.grad is not None
    assert not model.head.net[-1].weight.grad.any()
    torch.testing.assert_close(
        model.emg_model.featurizer[1].running_mean, before, rtol=0, atol=0
    )




def test_independent_heads_route_hands_and_mask_missing_emg(experts):
    model = BimanualSimpleResidualFusion(
        'vision', 'emg', independent_hand_heads=True).eval()
    assert isinstance(model.head, nn.ModuleList) and len(model.head) == 2
    assert not ({id(p) for p in model.head[0].parameters()}
                & {id(p) for p in model.head[1].parameters()})
    data = batch()
    with torch.no_grad():
        for side, head in enumerate(model.head):
            head.net[-1].weight.zero_()
            head.net[-1].bias.fill_(side+1.)
    pred = model(data)[0]
    expected = baseline(model, data)
    correction = (pred-expected).reshape(2, 2, 22, 1)
    torch.testing.assert_close(correction[:, 0], torch.ones(2, 22, 1))
    torch.testing.assert_close(correction[:, 1], torch.full((2, 22, 1), 2.))
    data['emg_valid_mask'][0, 1] = False
    data['emg'][0, 1] = float('nan')
    pred = model(data)[0]
    assert torch.isfinite(pred).all()
    torch.testing.assert_close(pred[1], expected[1], rtol=0, atol=0)
    model.zero_grad()
    pred[::2].sum().backward()
    assert model.head[0].net[-1].bias.grad.abs().sum() > 0
    assert all(p.grad is None or not p.grad.any() for p in model.head[1].parameters())


def test_independent_recipe_and_optimizer_coverage(monkeypatch):
    from egoemg.training.vision import ResidualFusionPredictionModule
    from egoemg.train import make_lightning_module

    vision, emg = Vision(), Emg()
    emg.head = nn.Conv1d(256, 22, 1)
    loaded = []

    def load(path):
        loaded.append(path)
        return vision if path == 'vision' else emg

    monkeypatch.setattr('egoemg.models.modules.bimanual_fusion.load_branch', load)
    with initialize_config_dir(
        version_base='1.3',
        config_dir=str(Path(__file__).resolve().parents[3] / 'config'),
    ):
        cfg = compose(config_name='base', overrides=[
            'experiment=fusion/fusion_vit_s'])
    assert cfg.module.independent_hand_heads
    assert cfg.module.emg_checkpoint.endswith('emgformer_s.ckpt')
    assert cfg.module.emg_config is None
    assert not cfg.module.freeze_vision and not cfg.module.freeze_emg
    assert cfg.optimizer.lr == 1e-4
    assert cfg.lr_scheduler.scheduler.eta_min == 1e-5
    assert cfg.trainer.max_epochs == 50 and cfg.trainer.check_val_every_n_epoch == 1
    assert {k for k, v in cfg.loss_weights.items() if v} == {'mae'}
    assert cfg.datamodule._target_ == "egoemg.training.datamodule.WindowedEmgDataModule"
    cfg.module.vision_checkpoint = 'vision'
    cfg.module.emg_checkpoint = 'emg'
    module = make_lightning_module(cfg)
    assert loaded == ['vision', 'emg']
    assert module.model.emg_model is emg
    assert all(not p.requires_grad for p in emg.head.parameters())
    assert isinstance(module, ResidualFusionPredictionModule)
    optimizer = module.configure_optimizers()['optimizer']
    assigned = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in module.parameters() if p.requires_grad}
    loss, *_ = module._gate_step(batch())
    loss.backward()
    for layer in [module.model.head[0], module.model.head[1],
                  module.model.vision, module.model.emg_model.featurizer,
                  module.model.temporal_attn, module.model.fusion_proj]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in layer.parameters())


def test_independent_heads_accept_vit_base_feature_width(monkeypatch):
    vision, emg = Vision(), Emg()
    vision.backbone[-1] = nn.Linear(3, 768)
    vision.head[0] = nn.Linear(768, 44)
    monkeypatch.setattr('egoemg.models.modules.bimanual_fusion.load_branch',
                        lambda path: vision if path == 'vision' else emg)
    model = BimanualSimpleResidualFusion(
        'vision', 'emg', independent_hand_heads=True,
        freeze_vision=False, freeze_emg=False)
    assert model.vision_proj.in_features == 768
    prediction, target, mask = model(batch())
    assert prediction.shape == target.shape == mask.shape == (4, 22, 1)
    (prediction-target).abs().mean().backward()
    assert model.vision_proj.weight.grad.abs().sum() > 0
