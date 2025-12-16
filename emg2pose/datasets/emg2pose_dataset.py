# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import KeysView
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch
from tqdm import tqdm

from emg2pose import transforms
from emg2pose.transforms import Transform
from emg2pose.utils import get_contiguous_ones, get_ik_failures_mask


@dataclass
class Emg2PoseSessionData:
    """Read-only interface to a single emg2pose session HDF5 file."""

    HDF5_GROUP: ClassVar[str] = "emg2pose"
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG: ClassVar[str] = "emg"
    JOINT_ANGLES: ClassVar[str] = "joint_angles"
    TIMESTAMPS: ClassVar[str] = "time"
    SESSION_NAME: ClassVar[str] = "session"
    SIDE: ClassVar[str] = "side"
    STAGE: ClassVar[str] = "stage"
    START_TIME: ClassVar[str] = "start"
    END_TIME: ClassVar[str] = "end"
    NUM_CHANNELS: ClassVar[str] = "num_channels"
    DATASET_NAME: ClassVar[str] = "dataset"
    USER: ClassVar[str] = "user"
    SAMPLE_RATE: ClassVar[str] = "sample_rate"

    hdf5_path: Path

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        emg2pose_group: h5py.Group = self._file[self.HDF5_GROUP]

        self.timeseries: h5py.Dataset = emg2pose_group[self.TIMESERIES]
        assert self.timeseries.dtype.fields is not None
        assert self.EMG in self.timeseries.dtype.fields
        assert self.JOINT_ANGLES in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields

        self.metadata: dict[str, Any] = {}
        for key, val in emg2pose_group.attrs.items():
            self.metadata[key] = val

    def __enter__(self) -> Emg2PoseSessionData:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()

    def __len__(self) -> int:
        return len(self.timeseries)

    def __getitem__(self, key: slice) -> np.ndarray:
        return self.timeseries[key]

    def slice(self, start_t: float = -np.inf, end_t: float = np.inf) -> np.ndarray:
        start_idx, end_idx = self.timestamps.searchsorted([start_t, end_t])
        return self[start_idx:end_idx]

    @property
    def fields(self) -> KeysView[str]:
        return self.timeseries.dtype.fields.keys()

    @property
    def timestamps(self) -> np.ndarray:
        emg_timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(emg_timestamps) >= 0).all(), "Not monotonic"
        return emg_timestamps

    @property
    def session_name(self) -> str:
        return self.metadata[self.SESSION_NAME]

    @property
    def user(self) -> str:
        return self.metadata[self.USER]

    @property
    def no_ik_failure(self):
        if not hasattr(self, "_no_ik_failure"):
            group = self._file[self.HDF5_GROUP]
            if "ik_failure_mask" in group:
                mask = ~group["ik_failure_mask"][...]
            else:
                joint_angles = self.timeseries[self.JOINT_ANGLES]
                mask = get_ik_failures_mask(joint_angles)

            self._no_ik_failure = np.asarray(mask, dtype=bool)
        return self._no_ik_failure

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.session_name} ({len(self)} samples)"


@dataclass
class WindowedEmgDataset(torch.utils.data.Dataset):
    """Iterates EMG windows with optional IK-failure filtering."""

    hdf5_path: Path
    window_length: InitVar[int | None] = 10_000
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(
        default_factory=transforms.ExtractToTensor
    )
    skip_ik_failures: bool = False

    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ) -> None:
        self.session_length = len(self.session)
        self.window_length = (
            window_length if window_length is not None else self.session_length
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        if window_length is None and self.skip_ik_failures:
            raise ValueError(
                "skip_ik_failures=True requires window_length to be specified."
            )

        self.windows: list[tuple[int, int]] = self.precompute_windows()

    def __len__(self) -> int:
        return sum(self._get_block_len(b) for b in self.blocks)

    def _get_block_len(self, block: tuple[float, float]) -> tuple[int]:
        return (block[1] - block[0] - self.window_length) // self.stride + 1

    @property
    def session(self):
        if not hasattr(self, "_session"):
            self._session = Emg2PoseSessionData(self.hdf5_path)
        return self._session

    @property
    def blocks(self) -> list[tuple[int, int]]:
        if not hasattr(self, "_blocks"):

            if not self.skip_ik_failures:
                if len(self.session) < self.window_length:
                    self._blocks = []
                else:
                    self._blocks = [(0, len(self.session))]
            else:
                blocks = get_contiguous_ones(self.session.no_ik_failure)
                blocks = [
                    (t0, t1 - 1)
                    for (t0, t1) in blocks
                    if (t1 - t0) >= self.window_length
                ]
                self._blocks = blocks

        return self._blocks

    def precompute_windows(self) -> list[tuple[int, int]]:
        windows = []
        cumsum = np.cumsum([0] + [self._get_block_len(b) for b in self.blocks])
        for idx in range(len(self)):
            block_idx = np.searchsorted(cumsum, idx, "right") - 1
            start_idx, end_idx = self.blocks[block_idx]
            relative_idx = idx - cumsum[block_idx]
            windows.append((start_idx + relative_idx * self.stride, end_idx))

        return windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        offset, end_idx = self.windows[idx]
        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds, leftover {leftover}")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]

        emg = self.transform(window)
        assert torch.is_tensor(emg)

        joint_angles = window[Emg2PoseSessionData.JOINT_ANGLES]
        joint_angles = torch.as_tensor(joint_angles)

        finite_mask = torch.isfinite(joint_angles).all(dim=1)  # T
        if not finite_mask.all():
            joint_angles = torch.nan_to_num(joint_angles, nan=0.0, posinf=0.0, neginf=0.0)

        mask = torch.as_tensor(self.session.no_ik_failure[window_start:window_end])
        mask = mask & finite_mask
        return {
            "emg": emg.T,  # CT
            "joint_angles": joint_angles.T,  # CT
            "label_valid_mask": mask,  # T
            "window_start_idx": window_start,
            "window_end_idx": window_end,
        }
