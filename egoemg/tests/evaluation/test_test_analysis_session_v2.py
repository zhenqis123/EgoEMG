"""Session-v2 test_analysis smoke; CLI: python -m egoemg.test_analysis
experiment=emgformer/regression_egoemg_session_v2 checkpoint=/path/to/model.ckpt
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from egoemg.test_analysis import EMG2PoseEvaluation, infer_dataset_type


class _Samples(Dataset):
    def __init__(self, hand, records):
        self.target_hand = hand
        self.records = records
        self._sessions = [
            SimpleNamespace(session_id="s1", source_group="recent", subject="0001"),
            SimpleNamespace(session_id="s2", source_group="showee", subject="0002"),
            SimpleNamespace(session_id="s3", source_group="original", subject="0003"),
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        session_id, finger_value, wrist_value = self.records[index]
        targets = torch.full((22, 2), finger_value)
        targets[20:] = wrist_value
        return {
            "emg": torch.zeros(8, 2),
            "joint_angles": targets,
            "label_valid_mask": torch.ones(22, 2, dtype=torch.bool),
            "episode_id": session_id,
            "held_out_user": False,
            "held_out_stage": True,
        }


class _ZeroModel(torch.nn.Module):
    def forward(self, batch):
        targets = batch["joint_angles"]
        return torch.zeros_like(targets), targets, batch["label_valid_mask"]


def test_test_loader_respects_configured_test_split_without_override(monkeypatch):
    evaluation = object.__new__(EMG2PoseEvaluation)
    evaluation.config = OmegaConf.create({"dataset": {"test": [{
        "_target_": "placeholder", "allowed_splits": ["recent_test"],
    }]}})
    evaluation.batch_size = 2
    evaluation.num_workers = 0
    seen = []

    def instantiate_test(config, **kwargs):
        seen.append(kwargs.get("allowed_splits", list(config.allowed_splits)))
        return _Samples("left", [("s1", 1.0, 2.0)])

    monkeypatch.setattr("egoemg.test_analysis.instantiate", instantiate_test)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaders = evaluation._get_session_v2_dataloaders()
    assert seen == [["recent_test"]]
    assert len(loaders) == 1
    evaluation.config.egoemg_test_splits = ["recent_user"]
    evaluation._get_session_v2_dataloaders()
    assert seen[-1] == ["recent_user"]


def test_bimanual_dataset_is_identified_as_session_archive():
    config = OmegaConf.create({"dataset": {"test": [{
        "_target_": "egoemg.datasets.session.fullframe.EgoEmgSessionFullFrameDataset",
    }]}})
    assert infer_dataset_type(config) == "egoemg_session_v2"


def test_bimanual_analysis_skips_masked_companion_hand(tmp_path, monkeypatch):
    class BimanualSamples(_Samples):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["joint_angles"] = torch.ones(44, 1)
            sample["joint_angles"][22:] = 3
            sample["label_valid_mask"] = torch.ones(44, 1, dtype=torch.bool)
            sample["label_valid_mask"][:22] = False
            sample["split_id"] = torch.tensor([0, 4])
            return sample

    class BimanualZero(_ZeroModel):
        bimanual_output = True

        def forward(self, batch):
            pred, target, mask = super().forward(batch)
            return pred.reshape(-1, 22, 1), target.reshape(-1, 22, 1), mask.reshape(-1, 22, 1)

    dataset = BimanualSamples("unused", [("s1", 0., 0.)])
    evaluation = object.__new__(EMG2PoseEvaluation)
    evaluation.module = BimanualZero()
    evaluation._session_v2_datasets = [dataset]
    evaluation.dataloaders = [DataLoader(dataset)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("egoemg.test_analysis._session_results_output_path",
                        lambda: str(tmp_path / "results.csv"))
    monkeypatch.setattr("egoemg.test_analysis._session_summary_output_path",
                        lambda: str(tmp_path / "summary.csv"))
    result = evaluation._evaluate_session_v2()
    assert len(result) == 1
    assert result.iloc[0]["split"] == "recent_object"
    assert result.iloc[0]["hand"] == "right"
    assert result.iloc[0]["test_mae"] == 3.


def test_session_v2_inference_and_pooled_metrics(tmp_path, monkeypatch):
    config = OmegaConf.create({
        "dataset": {"test": [{
            "_target_": "egoemg.datasets.session.emg.EgoEmgSessionDataset"
        }]},
        "egoemg_unified_memmap_dir": "legacy_path",
    })
    assert infer_dataset_type(config) == "egoemg_session_v2"

    left = _Samples("left", [("s1", 1.0, 2.0), ("s2", 3.0, 9.0)])
    right = _Samples("right", [("s1", 2.0, 4.0)])
    evaluation = object.__new__(EMG2PoseEvaluation)
    evaluation.module = _ZeroModel()
    evaluation._session_v2_datasets = [left, right]
    evaluation.dataloaders = [DataLoader(left, batch_size=2), DataLoader(right)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "session_results.csv"
    monkeypatch.setattr(
        "egoemg.test_analysis._session_results_output_path", lambda: str(output)
    )
    summary_output = tmp_path / "session_summary.csv"
    monkeypatch.setattr(
        "egoemg.test_analysis._session_summary_output_path",
        lambda: str(summary_output),
    )

    overall = evaluation._evaluate_session_v2()
    details = pd.read_csv(output)
    s1 = details[(details.session_id == "s1") & (details.hand == "pooled")].iloc[0]
    assert s1.windows == 2
    assert s1.valid_finger_values == 80
    assert s1.valid_wrist_values == 8
    np.testing.assert_allclose(s1.test_mae, (40 + 80 + 8 + 16) / 88)
    showee = details[(details.session_id == "s2") & (details.hand == "left")].iloc[0]
    assert showee.valid_wrist_values == 0
    assert np.isnan(showee.wrist_mae)
    np.testing.assert_allclose(showee.test_mae, 3.0)
    assert set(overall.hand) == {"left", "right"}
    summary = pd.read_csv(summary_output)
    assert len(summary) == 3
    assert summary.loc[summary.session_id == "s3", "status"].iloc[0] == (
        "no_evaluable_windows"
    )
    np.testing.assert_allclose(
        summary.loc[summary.session_id == "s1", "test_mae"].iloc[0], s1.test_mae
    )
