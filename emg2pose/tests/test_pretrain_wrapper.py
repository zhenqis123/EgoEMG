import torch
from torch.utils.data import Dataset

from emg2pose.datasets.pretrain_wrapper import PretrainWrapperDataset


class _DummyDataset(Dataset):
    def __init__(self, joint_channels: int) -> None:
        self.joint_channels = joint_channels

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        _ = idx
        t = 16
        return {
            "emg": torch.randn(16, t),
            "joint_angles": torch.randn(self.joint_channels, t),
            "label_valid_mask": torch.ones(t, dtype=torch.bool),
        }


def test_ninapro_index_8_is_masked_out_for_22d_angles() -> None:
    ds = PretrainWrapperDataset(
        base=_DummyDataset(joint_channels=22),
        name="ninapro",
        angle_dim=22,
    )
    sample = ds[0]
    angle_mask = sample["angle_mask"]

    assert angle_mask.shape[0] == 22
    assert not angle_mask[8].any()
    assert angle_mask[:8].all()
    assert angle_mask[9:].all()


def test_non_ninapro_keeps_index_8_supervision() -> None:
    ds = PretrainWrapperDataset(
        base=_DummyDataset(joint_channels=22),
        name="emg2pose",
        angle_dim=22,
    )
    sample = ds[0]
    angle_mask = sample["angle_mask"]

    assert angle_mask.shape[0] == 22
    assert angle_mask[8].all()
