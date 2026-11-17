"""Prevent shared recording session IDs from widening benchmark partitions."""

import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from egoemg.test_analysis import EMG2PoseEvaluation


def test_group_loaders_select_exact_record_indices_and_test_split(monkeypatch):
    seen = []

    class FakeDataset(Dataset):
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError(index)

    monkeypatch.setattr('egoemg.test_analysis.Emg2PoseDataset', FakeDataset)
    evaluation = object.__new__(EMG2PoseEvaluation)
    evaluation.config = OmegaConf.create({})
    evaluation.memmap_dir = '/unused'
    evaluation.window_length = 7790
    evaluation.skip_ik_failures = True
    evaluation.split = 'test'
    evaluation.batch_size = 2
    evaluation.num_workers = 0
    # Same session ID, but different side/record/group memberships.
    frame = pd.DataFrame([
        dict(session_index=3, session_id='shared', generalization='user'),
        dict(session_index=9, session_id='shared', generalization='user_stage'),
    ])
    evaluation.groupby = frame.groupby(['generalization'])
    evaluation.get_dataloaders()
    assert [row['allowed_session_indices'] for row in seen] == [[3], [9]]
    assert all(row['allowed_splits'] == ['test'] for row in seen)
    assert all('allowed_sessions' not in row for row in seen)


def test_npz_retains_global_record_indices_after_split_filter(tmp_path):
    path = tmp_path / 'metadata.npz'
    np.savez_compressed(
        path, users_user=np.array(['a', 'b']), stages_stage=np.array(['x']),
        splits_split=np.array(['train', 'val', 'test']),
        session_start_idx=np.array([0, 100, 200]),
        session_end_idx=np.array([100, 200, 300]),
        session_length=np.array([100, 100, 100]),
        session_session_id=np.array(['shared', 'shared', 'shared']),
        session_filename=np.array(['r0', 'r1', 'r2']),
        session_user_id=np.array([0, 0, 1]),
        session_stage_id=np.array([0, 0, 0]),
        session_split_id=np.array([0, 2, 2]),
        session_generalization=np.array(['train', 'stage', 'user_stage']))
    evaluation = object.__new__(EMG2PoseEvaluation)
    evaluation.config = OmegaConf.create({})
    evaluation.split = 'test'
    frame = evaluation._get_corpus_df_from_npz(str(path))
    assert frame.session_index.tolist() == [1, 2]
    assert frame.filename.tolist() == ['r1', 'r2']
