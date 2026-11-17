"""Val/test grouping retains cross user/action combinations and original train."""

import json

import numpy as np
import pytest

from egoemg.datasets.evaluation_partition import (
    EvaluationPartition,
    load_evaluation_partition,
)
from egoemg.datasets.session.emg import (EgoEmgSessionDataset)
from egoemg.datasets.session.fullframe import (EgoEmgSessionFullFrameDataset)
from egoemg.datasets.session.alignment import (build_alignment, dependency_signature)


def policy(tmp_path):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(dict(
        legacy=dict(mode='session', val_sessions=['sample'], test_sessions=['other']),
        recent=dict(val_objects=[8], test_objects=[0]))))
    return path


def test_session_assignment_keeps_all_gestures_and_both(tmp_path):
    p = EvaluationPartition(policy(tmp_path), 'val')
    split = np.array([0, 1, 2, 3, 4, 4, -1])
    objects = np.array([-1, -1, -1, -1, 8, 0, -1])
    for sid, expected in [('sample', [0, 1, 1, 1, 1, 2, -1]),
                          ('other', [0, 2, 2, 2, 1, 2, -1])]:
        result = p.classify('any-user', split, np.arange(7), objects, sid)
        np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError, match='absent from policy'):
        p.classify('new', split, split, objects, 'unknown')
    with pytest.raises(ValueError, match='training selection'):
        load_evaluation_partition(tmp_path / 'policy.json', 'val', ['train'])


def test_native_windows_respect_eval_roles_without_changing_train(tmp_path):
    from egoemg.tests.datasets.test_session_fusion_dataset import make_archive

    make_archive(tmp_path)
    path = policy(tmp_path)
    original = EgoEmgSessionDataset(tmp_path, 'left', ['gesture'], 4, 4)
    val = EgoEmgSessionDataset(tmp_path, 'left', ['gesture'], 4, 4,
                              evaluation_partition_path=path, evaluation_role='val')
    test = EgoEmgSessionDataset(tmp_path, 'left', ['gesture'], 4, 4,
                               evaluation_partition_path=path, evaluation_role='test')
    np.testing.assert_array_equal(original._starts, val._starts)
    assert len(test) == 0
    train = EgoEmgSessionDataset(tmp_path, 'left', ['train'], 4, 4)
    assert train._starts.tolist() == [0, 4, 8, 12]


def test_crop_roles_and_emg_windows_respect_object_boundary(tmp_path):
    from egoemg.datasets.session.fusion import EgoEmgSessionFusionDataset
    from egoemg.datasets.session.crops import (EgoEmgSessionVisionDataset)
    from egoemg.tests.datasets.test_session_fusion_dataset import make_archive

    stats = make_archive(tmp_path)
    folder = tmp_path / 'sample'
    session_path = folder / 'session.json'
    session = json.loads(session_path.read_text())
    arrays = session['hands']['left']['arrays']
    np.full(32, 4, np.int8).tofile(folder / arrays['split_id']['path'])
    objects = np.r_[np.full(16, 8), np.zeros(16)].astype(np.int32)
    objects.tofile(folder / 'objects.dat')
    arrays['object_id'] = dict(path='objects.dat', dtype='int32', shape=[32])
    session_path.write_text(json.dumps(session))
    common = dict(sample_interval_us=0, auto_build_index=True,
                  evaluation_partition_path=policy(tmp_path))
    val = EgoEmgSessionVisionDataset(
        tmp_path, ['recent_object'], evaluation_role='val', **common)
    test = EgoEmgSessionVisionDataset(
        tmp_path, ['recent_object'], evaluation_role='test', **common)
    assert val.samples[:, 2].tolist() == [2, 8, 14]
    assert test.samples[:, 2].tolist() == [18, 24, 30]
    assert not (set(val.samples[:, 1]) & set(test.samples[:, 1]))
    # Centers 14/18 are valid images, but their EMG windows cross roles.
    for role, expected in [('val', [8]), ('test', [24])]:
        fusion = EgoEmgSessionFusionDataset(
            tmp_path, ['recent_object'], window_length=8, norm_stats_path=stats,
            evaluation_role=role, **common)
        assert fusion.samples[:, 2].tolist() == expected


def test_fullframe_roles_filter_images_and_independent_hand_masks(tmp_path):
    folder = tmp_path / 'sample'
    folder.mkdir()

    def store(name, values):
        values = np.asarray(values)
        values.tofile(folder / name)
        return dict(path=name, shape=list(values.shape), dtype=str(values.dtype))

    times = store('times.dat', np.arange(4, dtype=np.int64) * 500000)
    hands = {}
    for h, side in enumerate(['left', 'right']):
        hands[side] = dict(arrays=dict(
            timestamps=times,
            split_id=store(side+'split.dat', np.array([4, 4, 4, 0], np.int8)),
            object_id=store(side+'objects.dat', np.array([8, 0, 8 if h == 0 else 0, -1])),
            joint_angles_valid=store(side+'valid.dat', np.ones(4, np.uint8)),
            joint_angles=store(side+'angles.dat', np.zeros((4, 20), np.float32)),
            wrist_angles=store(side+'wrists.dat', np.zeros((4, 2), np.float32)),
            wrist_valid=store(side+'wv.dat', np.ones(4, np.uint8))))
    (folder / 'video.mp4').touch()
    session = dict(session_id='sample', subject='a', source_group='recent', hands=hands,
                   cameras=dict(showee_head=dict(streams=dict(camera=dict(
                       arrays=dict(timestamps=times), video=dict(path='video.mp4'))))))
    path = folder / 'session.json'
    path.write_text(json.dumps(session))
    (tmp_path / 'dataset.json').write_text(json.dumps(dict(
        sessions=[dict(path='sample/session.json')])) )
    cache = tmp_path / 'fullframe_256_index'
    cache.mkdir()
    np.save(cache / 'sample.npy', np.zeros((4, 256, 256, 3), np.uint8))
    np.savez(cache / 'sample.npz', **build_alignment(folder, session))
    (cache / 'sample.json').write_text(json.dumps(dict(
        signature=dependency_signature(path, session, 400000))))
    kwargs = dict(evaluation_partition_path=policy(tmp_path))
    val = EgoEmgSessionFullFrameDataset(tmp_path, ['recent_object'],
                                       evaluation_role='val', **kwargs)
    test = EgoEmgSessionFullFrameDataset(tmp_path, ['recent_object'],
                                        evaluation_role='test', **kwargs)
    assert len(val) == len(test) == 1
    assert val[0]['object_id'].tolist() == [8, 8]
    assert test[0]['object_id'].tolist() == [0, 0]
    assert val[0]['label_valid_mask'].all() and test[0]['label_valid_mask'].all()
