"""Temporal RGB sampling, center-only targets, fast batches and model routing."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from egoemg.training.datamodule import WindowedEmgDataModule
from egoemg.datasets.session.fullframe import (EgoEmgSessionFullFrameDataset)
from egoemg.datasets.session.alignment import (build_alignment, dependency_signature)
from egoemg.datasets.session.temporal import (
    EgoEmgSessionTemporalVisionDataset, temporal_neighbors,
)
from egoemg.datasets.session_batch import SessionBatchDataset
from egoemg.models.modules.bimanual_temporal_vision import BimanualTemporalVisionPose
from egoemg.tests.models.test_bimanual_fusion import Vision


def test_neighbors_use_native_ordinals_and_do_not_cross_partitions_or_time_gaps():
    frames = np.arange(151)
    times = frames*33334
    allowed = np.ones(151, dtype=bool)
    allowed[65:90] = False
    rows, valid = temporal_neighbors(frames, times, allowed, np.array([45]), 30)
    assert valid.tolist() == [[False, True, True, False, False]]
    assert frames[rows].tolist() == [[45, 15, 45, 45, 45]]
    allowed[:] = True
    times[80:] += 1_000_000
    _, valid = temporal_neighbors(frames, times, allowed, np.array([70]), 30)
    assert not valid[0, 3:].any()
    # A 60fps camera cached at 30fps: native stride40 is cache stride20.
    frames = np.arange(0, 201, 2)
    rows, valid = temporal_neighbors(frames, frames*16667,
                                     np.ones(len(frames), bool), np.array([100]), 40)
    assert frames[rows].tolist() == [[20, 60, 100, 140, 180]]
    assert valid.all()


def test_seventeen_frames_cover_four_seconds_at_both_camera_rates():
    for fps in [30, 60]:
        # The dense cache retains 30fps even for a 60fps camera.
        times = np.rint(np.arange(151)*1_000_000/30).astype(np.int64)
        native_frames = np.arange(151)*(fps//30)
        rows, valid = temporal_neighbors(
            times, times, np.ones(len(times), bool),
            np.array([2_500_000]), 250_000, num_frames=17)
        assert valid.all()
        assert rows.shape == (1, 17)
        np.testing.assert_allclose(
            times[rows[0]], 2_500_000 + np.arange(-8, 9)*250_000,
            atol=17_000)
        assert native_frames[rows[0, 8]] == round(2.5*fps)
        assert times[rows[0, -1]]-times[rows[0, 0]] == 4_000_000


@pytest.fixture
def cached_session(tmp_path):
    folder = tmp_path/'sessions/sample'
    folder.mkdir(parents=True)
    n = 151
    def store(name, values):
        values = np.asarray(values)
        values.tofile(folder/name)
        return dict(path=name, shape=list(values.shape), dtype=str(values.dtype))
    times = store('times.dat', np.arange(n, dtype=np.int64)*33334)
    hands = {}
    for side in ['left', 'right']:
        hands[side] = dict(arrays=dict(
            timestamps=times,
            split_id=store(side+'_split.dat', np.zeros(n, np.int8)),
            joint_angles_valid=store(side+'_valid.dat', np.ones(n, np.uint8)),
            joint_angles=store(side+'_angles.dat', np.repeat(
                np.arange(n, dtype=np.float32)[:, None], 20, 1)),
            wrist_angles=store(side+'_wrist.dat', np.zeros((n, 2), np.float32)),
            wrist_valid=store(side+'_wv.dat', np.ones(n, np.uint8))))
    (folder/'video.mp4').touch()
    session = dict(session_id='sample', subject='person', source_group='recent',
        hands=hands, cameras=dict(showee_head=dict(streams=dict(camera=dict(
            arrays=dict(timestamps=times), video=dict(path='video.mp4'))))))
    path = folder/'session.json'
    path.write_text(json.dumps(session))
    (tmp_path/'dataset.json').write_text(json.dumps(dict(sessions=[
        dict(path='sessions/sample/session.json')])) )
    sparse, dense = tmp_path/'sparse', tmp_path/'dense'
    for directory, interval, fps in [(sparse, 400000, None), (dense, 0, 30)]:
        directory.mkdir()
        alignment = build_alignment(folder, session, interval, fps)
        frames = alignment['frames']
        rgb = np.broadcast_to(frames[:, None, None, None], (len(frames), 4, 4, 3))
        np.save(directory/'sample.npy', rgb.astype(np.uint8))
        np.savez(directory/'sample.npz', **alignment)
        (directory/'sample.json').write_text(json.dumps(dict(
            signature=dependency_signature(path, session, interval, fps))))
    return tmp_path, sparse, dense


def test_temporal_cache_preserves_sparse_centers_labels_and_fast_batch(cached_session):
    root, sparse, dense = cached_session
    base = EgoEmgSessionFullFrameDataset(root, ['train'], cache_dir=sparse)
    ds = EgoEmgSessionTemporalVisionDataset(root, ['train'], cache_dir=sparse,
                                          context_cache_dir=dense)
    np.testing.assert_array_equal(ds.samples, base.samples)
    for i in [0, 5, len(ds)-1]:
        a, b = base[i], ds[i]
        assert b['vision_img'].shape == (5, 3, 4, 4)
        np.testing.assert_array_equal(b['vision_img'][2], a['vision_img'])
        np.testing.assert_array_equal(b['joint_angles'], a['joint_angles'])
        np.testing.assert_array_equal(b['label_valid_mask'], a['label_valid_mask'])
        for slot in np.flatnonzero(~b['temporal_valid_mask']):
            np.testing.assert_array_equal(b['vision_img'][slot], a['vision_img'])
    sample = ds[5]
    assert sample['temporal_camera_frames'].tolist() == [0, 30, 60, 90, 120]
    assert sample['temporal_valid_mask'].all()
    fast = SessionBatchDataset(ds, WindowedEmgDataModule._collate_fn)
    actual = fast.__getitems__([0, 5, len(ds)-1])
    expected = WindowedEmgDataModule._collate_fn([ds[i] for i in [0, 5, len(ds)-1]])
    for key in ['vision_img', 'joint_angles', 'label_valid_mask', 'temporal_valid_mask',
                'temporal_camera_frames']:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def model_and_batch(monkeypatch):
    monkeypatch.setattr('egoemg.models.modules.bimanual_temporal_vision.load_branch',
                        lambda path: Vision())
    model = BimanualTemporalVisionPose('unused', hidden_dim=32, heads=4, dropout=0.)
    batch = dict(vision_img=torch.rand(2, 5, 3, 8, 8),
                 temporal_valid_mask=torch.ones(2, 5, dtype=torch.bool),
                 joint_angles=torch.randn(2, 44, 1),
                 label_valid_mask=torch.ones(2, 44, 1, dtype=torch.bool))
    return model, batch


def test_zero_initialization_reuses_center_head_and_learns_temporal(monkeypatch):
    model, batch = model_and_batch(monkeypatch)
    model.eval()
    expected = model.vision.head(model.vision.backbone(batch['vision_img'][:, 2]))
    prediction, target, mask = model(batch)
    torch.testing.assert_close(prediction, expected.reshape(-1, 22, 1))
    assert prediction.shape == target.shape == mask.shape == (4, 22, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    (prediction-target).abs().mean().backward()
    assert model.temporal.output.weight.grad.abs().sum() > 0
    opt.step();opt.zero_grad()
    (model(batch)[0]-target).abs().mean().backward()
    assert model.temporal.input.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.parameters())


def test_invalid_context_cannot_change_center_prediction(monkeypatch):
    model, batch = model_and_batch(monkeypatch)
    model.eval()
    torch.nn.init.normal_(model.temporal.output.weight, std=.05)
    batch['temporal_valid_mask'][:, [0, 4]] = False
    reference = model(batch)[0].detach()
    changed = copy.deepcopy(batch)
    changed['vision_img'][:, [0, 4]] = 100.
    torch.testing.assert_close(model(changed)[0], reference)
    changed['vision_img'][:, 1] = 100.
    assert not torch.allclose(model(changed)[0], reference)




def test_bf16_strategy_uses_native_hook_without_touching_other_trainers(monkeypatch):
    from omegaconf import OmegaConf
    import egoemg.training.runtime as utils
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
        bf16_compress_hook,
    )
    captured = {}
    def fake_trainer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)
    monkeypatch.setattr(utils.pl, 'Trainer', fake_trainer)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    c = OmegaConf.create(dict(bf16_gradient_communication=True,
        trainer=dict(devices=[0, 1], strategy='ddp')))
    t = utils.build_trainer(c, [], False)
    assert t.strategy._ddp_comm_hook is bf16_compress_hook
    c.bf16_gradient_communication = False
    t = utils.build_trainer(c, [], False)
    assert t.strategy == 'ddp'


def test_seventeen_frame_cache_and_random_head_initialization(cached_session):
    root, sparse, dense = cached_session
    ds = EgoEmgSessionTemporalVisionDataset(
        root, ['train'], cache_dir=sparse, context_cache_dir=dense,
        num_frames=17, frame_interval_us=250000)
    base = EgoEmgSessionFullFrameDataset(root, ['train'], cache_dir=sparse)
    for i in [0, len(ds)//2, len(ds)-1]:
        sample = ds[i]
        assert sample['vision_img'].shape == (17, 3, 4, 4)
        np.testing.assert_array_equal(sample['vision_img'][8], base[i]['vision_img'])
        np.testing.assert_array_equal(sample['joint_angles'], base[i]['joint_angles'])
        assert sample['temporal_valid_mask'][8]
    assert not ds[0]['temporal_valid_mask'][:8].any()
    model = BimanualTemporalVisionPose(vision_config=Vision(), num_frames=17)
    model.eval()
    images = torch.randn(2, 17, 3, 4, 4)
    valid = torch.ones(2, 17, dtype=torch.bool)
    result = model(dict(vision_img=images, temporal_valid_mask=valid))
    expected = model.vision.head(model.vision.backbone(images[:, 8]))
    torch.testing.assert_close(result, expected.reshape(-1, 22, 1))
    assert all(p.requires_grad for p in model.parameters())


def test_seventeen_frame_config_uses_uniform_lr_and_no_early_stop():
    with initialize_config_dir(version_base='1.3', config_dir=str(
            Path(__file__).resolve().parents[3]/'config')):
        c = compose(config_name='base', overrides=[
            'experiment=fusion/temporal_vision_vit_s'])
    assert c.module.vision_checkpoint is None
    assert c.module.vision_config.pretrained
    assert c.module.vision_config.backbone_type == 'vit_small'
    assert c.module.num_frames == 17
    assert c.dataset.train[0].frame_interval_us == 250000
    assert c.dataset.val[0].num_frames == c.dataset.test[0].num_frames == 17
    assert c.component_lr_scales.vision == c.component_lr_scales.temporal == 1.0
    assert c.optimizer.lr == 5e-5
    assert c.lr_scheduler.scheduler.eta_min == 5e-6
    assert c.trainer.max_epochs == c.lr_scheduler.scheduler.T_max == 200
    assert all('EarlyStopping' not in x._target_ for x in c.callbacks)
