from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


def _pimforce_to_emg2pose_angles(
    pose: np.ndarray, *, pose_in_degrees: bool
) -> np.ndarray:
    """
    Convert PiMforce angle ordering to emg2pose joint ordering.

    PiMforce order: per finger [spread, flex1, flex2, flex3]
    emg2pose order: see emg2pose/constants.py
    """
    pose = pose.copy()
    # Match PiMforce kinematics: thumb CMC angles are offset in the raw data.
    if pose_in_degrees:
        pose[0] -= 20.0  # thumb spread/AA
        pose[1] -= 60.0  # thumb flexion/FE
    else:
        pose[0] -= np.deg2rad(20.0)
        pose[1] -= np.deg2rad(60.0)
    if pose_in_degrees:
        pose = np.deg2rad(pose)

    # Only thumb swaps AA/FE order; other fingers are already spread->AA then flex.
    idx_map = np.array(
        [
            1, 0, 2, 3,  # thumb: FE, AA, FE, FE
            4, 5, 6, 7,  # index: AA, FE, FE, FE
            8, 9, 10, 11,  # middle
            12, 13, 14, 15,  # ring
            16, 17, 18, 19,  # pinky
        ],
        dtype=np.int64,
    )
    return pose[idx_map, ...]


def _resolve_window(
    total_len: int,
    valid_len: int | None,
    *,
    window_start: int | None,
    window_stop: int | None,
    window_stride: int,
    clip_to_valid: bool,
) -> tuple[int, int, int]:
    if window_stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {window_stride}")
    start = 0 if window_start is None else window_start
    stop = window_stop
    if start < 0:
        start = max(total_len + start, 0)
    if stop is None or stop <= 0:
        stop = total_len
    elif stop < 0:
        stop = max(total_len + stop, 0)
    if clip_to_valid and valid_len is not None:
        stop = min(stop, valid_len)
    start = min(max(start, 0), total_len)
    stop = min(max(stop, 0), total_len)
    if stop < start:
        stop = start
    return start, stop, window_stride


def _slice_time(sample: np.ndarray, start: int, stop: int, stride: int) -> np.ndarray:
    if sample.ndim <= 1:
        return sample
    if start == 0 and stop == sample.shape[1] and stride == 1:
        return sample
    return sample[:, start:stop:stride]


def _valid_window_length(
    valid_len: int | None, start: int, stop: int, stride: int
) -> int | None:
    if valid_len is None:
        return None
    valid_stop = min(valid_len, stop)
    span = max(valid_stop - start, 0)
    return (span + stride - 1) // stride


class PiMforceSessionData:
    def __init__(
        self,
        root_dir: Path,
        emg_file: str = "emg_train.npy",
        lengths_file: str | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.emg_path = self.root_dir / emg_file
        self.emg = np.load(self.emg_path, mmap_mode="r")
        lengths_name = (
            lengths_file
            if lengths_file is not None
            else f"{Path(emg_file).stem}_lengths.npy"
        )
        self.lengths_path = self.root_dir / lengths_name
        self.lengths: np.ndarray | None = None
        if self.lengths_path.exists():
            self.lengths = np.load(self.lengths_path)
            if self.lengths.ndim != 1:
                raise ValueError(
                    f"Expected 1D lengths array at {self.lengths_path}, "
                    f"got {self.lengths.shape}"
                )
            if len(self.lengths) != len(self.emg):
                raise ValueError(
                    f"Lengths size {len(self.lengths)} does not match "
                    f"{len(self.emg)} samples in {self.emg_path}"
                )

    def __len__(self) -> int:
        return len(self.emg)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.emg[idx]


@dataclass
class WindowedPiMforceDataset(Dataset):
    root_dir: Path
    emg_file: str = "emg_train.npy"
    lengths_file: str | None = None
    window_start: int | None = 0
    window_stop: int | None = None
    window_stride: int = 1
    clip_to_valid: bool = False
    emg_channels: int = 8
    pose_channels: int = 20
    pose_mode: str = "last"  # last | sequence
    repeat_pose: bool = True
    pose_in_degrees: bool = True
    transform: Callable[[dict[str, np.ndarray]], Any] | None = None

    def __post_init__(self) -> None:
        self.session = PiMforceSessionData(
            self.root_dir, emg_file=self.emg_file, lengths_file=self.lengths_file
        )
        if self.pose_mode not in {"last", "sequence"}:
            raise ValueError(f"Unsupported pose_mode={self.pose_mode!r}")

    def __len__(self) -> int:
        return len(self.session)

    def _extract_emg_pose(
        self, sample: np.ndarray, valid_len: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        emg = sample[: self.emg_channels]
        pose = sample[self.emg_channels : self.emg_channels + self.pose_channels]
        if self.pose_mode == "last":
            last_idx = -1
            if valid_len is not None and pose.ndim > 1:
                last_idx = max(min(valid_len, pose.shape[1]), 1) - 1
            pose = pose[:, last_idx][:, None]
            if self.repeat_pose and emg.ndim == 2:
                pose = np.repeat(pose, emg.shape[1], axis=1)
        return emg, pose

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.session[idx]
        valid_len = None
        if self.session.lengths is not None:
            valid_len = int(self.session.lengths[idx])
        total_len = int(sample.shape[-1]) if sample.ndim > 1 else 1
        start, stop, stride = _resolve_window(
            total_len,
            valid_len,
            window_start=self.window_start,
            window_stop=self.window_stop,
            window_stride=self.window_stride,
            clip_to_valid=self.clip_to_valid,
        )
        sample = _slice_time(sample, start, stop, stride)
        valid_len_window = _valid_window_length(valid_len, start, stop, stride)
        emg, pose = self._extract_emg_pose(sample, valid_len=valid_len_window)
        joint_angles = _pimforce_to_emg2pose_angles(
            pose, pose_in_degrees=self.pose_in_degrees
        )

        if self.transform is not None:
            transformed = self.transform({"emg": emg, "joint_angles": joint_angles})
            if isinstance(transformed, tuple) and len(transformed) == 2:
                emg, joint_angles = transformed
            elif isinstance(transformed, dict):
                emg = transformed.get("emg", emg)
                joint_angles = transformed.get("joint_angles", joint_angles)
            else:
                emg = transformed

        emg_t = torch.as_tensor(emg)
        joint_angles_t = torch.as_tensor(joint_angles)
        time_len = int(emg_t.shape[-1]) if emg_t.ndim > 1 else 1
        mask = torch.zeros(time_len, dtype=torch.bool)
        if valid_len_window is None:
            mask[:] = True
        else:
            mask[: min(valid_len_window, time_len)] = True
        
        if valid_len_window is None:
            window_end = time_len
        else:
            window_end = min(valid_len_window, time_len)
        return {
            "emg": emg_t,  # CT
            "joint_angles": joint_angles_t,  # CT
            "label_valid_mask": mask,  # T (valid steps)
            "window_start_idx": 0,
            "window_end_idx": window_end,
        }


@dataclass
class MultiSessionWindowedPiMforceDataset(Dataset):
    root_dirs: list[Path]
    emg_file: str = "emg_train.npy"
    lengths_file: str | None = None
    window_start: int | None = 0
    window_stop: int | None = None
    window_stride: int = 1
    clip_to_valid: bool = False
    emg_channels: int = 8
    pose_channels: int = 20
    pose_mode: str = "last"
    repeat_pose: bool = True
    pose_in_degrees: bool = True
    transform: Callable[[dict[str, np.ndarray]], Any] | None = None

    def __post_init__(self) -> None:
        self.sessions = [
            PiMforceSessionData(
                Path(root), emg_file=self.emg_file, lengths_file=self.lengths_file
            )
            for root in self.root_dirs
        ]
        self._lengths = np.array([len(s) for s in self.sessions], dtype=np.int64)
        self._cumsum = np.cumsum(np.insert(self._lengths, 0, 0))
        if self.pose_mode not in {"last", "sequence"}:
            raise ValueError(f"Unsupported pose_mode={self.pose_mode!r}")

    def __len__(self) -> int:
        return int(self._cumsum[-1])

    def _extract_emg_pose(
        self, sample: np.ndarray, valid_len: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        emg = sample[: self.emg_channels]
        pose = sample[self.emg_channels : self.emg_channels + self.pose_channels]
        if self.pose_mode == "last":
            last_idx = -1
            if valid_len is not None and pose.ndim > 1:
                last_idx = max(min(valid_len, pose.shape[1]), 1) - 1
            pose = pose[:, last_idx][:, None]
            if self.repeat_pose and emg.ndim == 2:
                pose = np.repeat(pose, emg.shape[1], axis=1)
        return emg, pose

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        session_idx = int(np.searchsorted(self._cumsum, idx, side="right") - 1)
        local_idx = int(idx - self._cumsum[session_idx])
        sample = self.sessions[session_idx][local_idx]
        valid_len = None
        lengths = self.sessions[session_idx].lengths
        if lengths is not None:
            valid_len = int(lengths[local_idx])
        total_len = int(sample.shape[-1]) if sample.ndim > 1 else 1
        start, stop, stride = _resolve_window(
            total_len,
            valid_len,
            window_start=self.window_start,
            window_stop=self.window_stop,
            window_stride=self.window_stride,
            clip_to_valid=self.clip_to_valid,
        )
        sample = _slice_time(sample, start, stop, stride)
        valid_len_window = _valid_window_length(valid_len, start, stop, stride)
        emg, pose = self._extract_emg_pose(sample, valid_len=valid_len_window)
        joint_angles = _pimforce_to_emg2pose_angles(
            pose, pose_in_degrees=self.pose_in_degrees
        )

        if self.transform is not None:
            transformed = self.transform({"emg": emg, "joint_angles": joint_angles})
            if isinstance(transformed, tuple) and len(transformed) == 2:
                emg, joint_angles = transformed
            elif isinstance(transformed, dict):
                emg = transformed.get("emg", emg)
                joint_angles = transformed.get("joint_angles", joint_angles)
            else:
                emg = transformed

        emg_t = torch.as_tensor(emg)
        joint_angles_t = torch.as_tensor(joint_angles)
        time_len = int(emg_t.shape[-1]) if emg_t.ndim > 1 else 1
        mask = torch.zeros(time_len, dtype=torch.bool)
        if valid_len_window is None:
            mask[:] = True
        else:
            mask[: min(valid_len_window, time_len)] = True

        if valid_len_window is None:
            window_end = time_len
        else:
            window_end = min(valid_len_window, time_len)
        return {
            "emg": emg_t,  # CT
            "joint_angles": joint_angles_t,  # CT
            "label_valid_mask": mask,  # T (valid steps)
            "window_start_idx": 0,
            "window_end_idx": window_end,
        }
