"""Native full-window metrics and concise final-checkpoint reports."""
from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd
import torch
from omegaconf import OmegaConf

from egoemg.evaluate import Report, accumulate_batch, model_config
from egoemg.test_analysis import infer_dataset_type


def test_saved_emg_width_references_resolve_and_pretraining_is_disabled(tmp_path):
    checkpoint = {'hyper_parameters': {'module_conf': OmegaConf.create({
        '_target_': 'example', 'decoder': {'model_dim': 256},
        'head': {'in_channels': '${module.decoder.model_dim}'},
        'pretrained': True})}}
    conf = model_config(checkpoint, 'emgformer_s', tmp_path)
    assert conf.head.in_channels == 256
    assert conf.pretrained is False


def test_full_window_time_varying_groups_and_valid_angle_weighting(tmp_path):
    class Metadata:
        dataset = SimpleNamespace(target_hand='left')

        def row(self, batch, index, side, length, left_context):
            assert length == 4
            return 'session', 'user', 'egoemg', {
                'object_id': np.array([0, 0, 1, 1]),
                'gesture_class': np.array([2, 2, 3, 3])}

    model = SimpleNamespace(model=SimpleNamespace(left_context=0))
    prediction = torch.tensor([1., 1., 3., 3.])[None, None].expand(1, 22, 4)
    target = torch.zeros_like(prediction)
    valid = torch.ones_like(prediction, dtype=torch.bool)
    valid[:, 20:] = False
    report = Report()
    accumulate_batch(report, prediction, target, valid,
                     {'episode_id': ['session'], 'split_id': torch.tensor([4])},
                     Metadata(), model)
    assert report.splits['all', 'both'].metrics()['mae'] == 2.
    assert report.splits['all', 'both'].metrics()['valid_angles'] == 80
    assert report.splits['all', 'both'].metrics()['valid_timepoints'] == 4
    assert report.groups['object', '0', 'all', 'both'].metrics()['mae'] == 1.
    assert report.groups['object', '1', 'all', 'both'].metrics()['mae'] == 3.
    report.write(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        'results.csv', 'joint_results.csv', 'group_results.csv'}
    joints = pd.read_csv(tmp_path/'joint_results.csv')
    assert set(joints.joint) >= {'WRIST_FE', 'THUMB_CMC_FE'}


def test_bimanual_showee_wrist_mask_and_hand_roles():
    class Metadata:
        def row(self, batch, index, side, length, left_context):
            return 'session', 'user', 'showee', {}

    module = SimpleNamespace(
        model=SimpleNamespace(left_context=0, bimanual_output=True))
    prediction = torch.ones(2, 22, 1)
    target = torch.zeros_like(prediction)
    valid = torch.ones_like(prediction, dtype=torch.bool)
    report = Report()
    accumulate_batch(report, prediction, target, valid,
                     {'episode_id': ['session'], 'split_id': torch.tensor([[1, 4]])},
                     Metadata(), module)
    assert report.splits['all', 'both'].metrics()['valid_angles'] == 40
    assert report.splits['user', 'left'].metrics()['mae'] == 1.
    assert report.splits['recent_object', 'right'].metrics()['mae'] == 1.
    assert report.splits['all', 'both'].metrics()['samples'] == 2


def test_temporal_datasets_are_recognized_as_session_family():
    for kind in ['vision', 'fusion']:
        conf = OmegaConf.create({'dataset': {'test': [{
            '_target_': 'egoemg.datasets.egoemg_session_temporal_'+kind+'_dataset.'
                        'EgoEmgSessionTemporalDataset'}]}})
        assert infer_dataset_type(conf) == 'egoemg_session_v2'


def test_raw_variants_use_raw_training_statistics(monkeypatch, tmp_path):
    from egoemg.evaluate import datasets_for
    calls = []
    monkeypatch.setattr('egoemg.evaluate.EgoEmgSessionDataset',
                        lambda **kw: calls.append(kw))
    args = SimpleNamespace(dataset_dir=tmp_path, partition=tmp_path/'v6.json')
    checkpoint = {'hyper_parameters': {'datamodule': {
        'val_test_window_length': 8000, 'val_test_stride': 8000}}}
    datasets_for('emgformer_m_raw_aug', 'test', checkpoint, args)
    assert len(calls) == 2
    assert calls[0]['emg_field'] == 'emg_raw'
    assert calls[0]['norm_stats_path'].name == 'egoemg_emg_normalization_raw.json'
    assert calls[0]['evaluation_role'] == 'test'


def test_temporal_global_interpolations_resolve_without_training_config(tmp_path):
    checkpoint = {'hyper_parameters': {'module_conf': OmegaConf.create({
        '_target_': 'example', 'vision_checkpoint': '${temporal_vision_checkpoint}',
        'num_frames': '${temporal_num_frames}',
        'vision_config': {'pretrained': True}})}}
    conf = model_config(checkpoint, 'vision_vit_s_temporal_17f', tmp_path)
    assert conf.vision_checkpoint is None
    assert conf.num_frames == 17
    assert conf.vision_config.pretrained is False


@pytest.mark.parametrize('override', [False, True])
def test_cli_partition_follows_external_dataset_directory(monkeypatch, tmp_path, override):
    from egoemg.evaluate import main
    checkpoint_dir = tmp_path / 'weights'
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'emgformer_s.ckpt').touch()
    dataset = tmp_path / 'downloaded_dataset'
    expected = tmp_path / 'custom_protocol.json' if override else dataset / 'assets/egoemg_evaluation_protocol.json'
    seen = []
    monkeypatch.setattr('egoemg.evaluate.evaluate_one',
                        lambda path, role, args, cache: seen.append(args.partition))
    args = ['--checkpoint-dir', str(checkpoint_dir), '--dataset-dir', str(dataset),
            '--output-dir', str(tmp_path / 'results'), '--device', 'cpu']
    if override:
        args.extend(['--partition', str(expected)])
    main(args)
    assert seen == [expected]
