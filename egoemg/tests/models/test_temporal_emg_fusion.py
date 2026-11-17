"""Temporal checkpoint restoration, residual fallback and optimizer coverage."""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egoemg.training.vision import ResidualFusionPredictionModule
from egoemg.models.modules.bimanual_fusion import load_branch
from egoemg.models.modules.bimanual_temporal_vision import BimanualTemporalVisionPose
from egoemg.tests.models.test_bimanual_fusion import Emg, Vision
from egoemg.tests.models.test_temporal_vision import cached_session  # noqa: F401
from egoemg.train import make_lightning_module


def temporal_model():
    m = BimanualTemporalVisionPose(
        vision_config=Vision(), num_frames=17, hidden_dim=16, heads=4,
        dropout=0.)
    # Emulate a trained temporal projection, rather than the identity start.
    torch.nn.init.normal_(m.temporal.output.weight, std=.02)
    return m


def test_temporal_branch_restores_without_reinitializing_pose_head(tmp_path):
    m = temporal_model().eval()
    conf = OmegaConf.create(dict(
        _target_='egoemg.models.modules.bimanual_temporal_vision.'
                  'BimanualTemporalVisionPose',
        vision_config=dict(_target_='egoemg.tests.models.test_bimanual_fusion.Vision'),
        num_frames=17, hidden_dim=16, heads=4, dropout=0.))
    p = tmp_path/'temporal.ckpt'
    torch.save(dict(hyper_parameters=dict(module_conf=conf), state_dict={
        'model.'+k: v for k, v in m.state_dict().items()}), p)
    restored = load_branch(p).eval()
    batch = dict(vision_img=torch.randn(2, 17, 3, 4, 4),
                 temporal_valid_mask=torch.ones(2, 17, dtype=torch.bool))
    torch.testing.assert_close(m(batch), restored(batch), rtol=0, atol=0)
    assert restored.head is restored.vision.head
    assert not any(k.startswith('head.') for k in restored.state_dict())


def test_joint_temporal_fusion_gradients_fallback_and_uniform_lr(monkeypatch):
    vision, emg = temporal_model(), Emg()
    monkeypatch.setattr('egoemg.models.modules.bimanual_fusion.load_branch',
                        lambda path: vision if path == 'temporal' else emg)
    with initialize_config_dir(version_base='1.3', config_dir=str(
            Path(__file__).resolve().parents[3]/'config')):
        c = compose(config_name='base', overrides=[
            'experiment=fusion/temporal_fusion_vit_s',
            'fusion_vision_checkpoint=temporal'])
    module = make_lightning_module(c)
    assert isinstance(module, ResidualFusionPredictionModule)
    model = module.model.eval()
    batch = dict(vision_img=torch.randn(2, 17, 3, 4, 4),
                 temporal_valid_mask=torch.ones(2, 17, dtype=torch.bool),
                 emg=torch.randn(2, 2, 8, 20),
                 emg_valid_mask=torch.tensor([[True, False], [True, True]]),
                 joint_angles=torch.randn(2, 44, 1),
                 label_valid_mask=torch.ones(2, 44, 1, dtype=torch.bool))
    batch['emg'][0, 1] = float('nan')
    expected = vision(batch)[0]
    prediction, target, valid = model(batch)
    assert prediction.shape == target.shape == valid.shape == (4, 22, 1)
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction[1], expected[1], rtol=0, atol=0)
    model.force_zero_correction = True
    torch.testing.assert_close(model(batch)[0], expected, rtol=0, atol=0)
    model.force_zero_correction = False
    model.train()
    prediction, target, _ = model(batch)
    (prediction-target).abs().mean().backward()
    for layer in [vision.vision.backbone, vision.head, vision.temporal.attention,
                  emg.featurizer, emg.decoder, model.temporal_attn,
                  model.vision_proj, model.fusion_proj, model.head]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in layer.parameters())
    groups = module.configure_optimizers()['optimizer'].param_groups
    assert {g['name']: g['lr'] for g in groups} == {
        'vision': 1e-4, 'emg': 1e-4, 'fusion': 1e-4}
    assigned = [id(p) for g in groups for p in g['params']]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
    assert c.lr_scheduler.scheduler.eta_min == 1e-5
    assert c.trainer.max_epochs == c.lr_scheduler.scheduler.T_max == 100
    assert c.callbacks[-1].patience == 20
    assert c.datamodule.train_samples_per_epoch == 33786
    assert c.dataset.train[0].num_frames == 17
    assert c.dataset.train[0].jitter is False


def test_temporal_emg_shared_batch_matches_single_reader(cached_session):  # noqa: F811
    import json
    import numpy as np
    from egoemg.training.datamodule import WindowedEmgDataModule
    from egoemg.datasets.session.alignment import (dependency_signature)
    from egoemg.datasets.session.fusion import (
        EgoEmgSessionFullFrameFusionDataset,
    )
    from egoemg.datasets.session.fusion import (
        EgoEmgSessionTemporalFusionDataset,
    )
    from egoemg.datasets.session_batch import SessionBatchDataset

    root, sparse, dense = cached_session
    folder = root/'sessions/sample'
    session_path = folder/'session.json'
    session = json.loads(session_path.read_text())
    stats = {}
    for i, side in enumerate(['left', 'right']):
        values = np.arange(151*8, dtype=np.float32).reshape(151, 8)+i
        name = side+'_emg.dat'
        values.tofile(folder/name)
        session['hands'][side]['arrays']['emg_filtered'] = dict(
            path=name, shape=[151, 8], dtype='float32', unit='mV')
        stats['emg_filtered_'+side] = dict(mean=[0.]*8, std=[1.]*8)
    session_path.write_text(json.dumps(session))
    stats_path = root/'stats.json'
    stats_path.write_text(json.dumps(stats))
    for directory, interval, fps in [(sparse, 400000, None), (dense, 0, 30)]:
        (directory/'sample.json').write_text(json.dumps(dict(signature=
            dependency_signature(session_path, session, interval, fps))))
    options = dict(dataset_dir=root, allowed_splits=['train'], cache_dir=sparse,
                   window_length=8, norm_stats_path=stats_path, jitter=False)
    ds = EgoEmgSessionTemporalFusionDataset(
        **options, context_cache_dir=dense, num_frames=17, frame_interval_us=250000)
    base = EgoEmgSessionFullFrameFusionDataset(**options)
    sample = ds[5]
    assert sample['vision_img'].shape == (17, 3, 4, 4)
    assert sample['emg_valid_mask'].all()
    np.testing.assert_array_equal(sample['vision_img'][8], base[5]['vision_img'])
    np.testing.assert_array_equal(sample['emg'], base[5]['emg'])
    np.testing.assert_array_equal(sample['window_offset_samples'], [0, 0])
    indices = [0, 5, len(ds)-1]
    actual = SessionBatchDataset(ds, WindowedEmgDataModule._collate_fn).__getitems__(
        indices)
    expected = WindowedEmgDataModule._collate_fn([ds[i] for i in indices])
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]
