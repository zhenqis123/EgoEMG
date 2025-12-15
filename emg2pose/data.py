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

from emg2pose import transforms
from emg2pose.transforms import Transform
from emg2pose.utils import get_contiguous_ones, get_ik_failures_mask
from tqdm import tqdm


@dataclass
class Emg2PoseSessionData:
    """A read-only interface to a single emg2pose session file stored in
    HDF5 format.

    ``self.timeseries`` is a `h5py.Dataset` instance with a compound data type
    as in a numpy structured array containing three fields - EMG data from the
    left and right wrists, and their corresponding timestamps.
    The sampling rate of EMG is 2kHz, each EMG device has 16 electrode
    channels, and the signal has been high-pass filtered. Therefore, the fields
    corresponding to left and right EMG are 2D arrays of shape ``(T, 16)`` each
    and ``timestamps`` is a 1D array of length ``T``.

    NOTE: Only the metadata and ground-truth are loaded into memory while the
    EMG data is accesssed directly from disk. When wrapping this interface
    within a PyTorch Dataset, use multiple dataloading workers to mask the
    disk seek and read latencies."""

    HDF5_GROUP: ClassVar[str] = "emg2pose"
    # timeseries keys
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG: ClassVar[str] = "emg"
    JOINT_ANGLES: ClassVar[str] = "joint_angles"
    TIMESTAMPS: ClassVar[str] = "time"
    # metadata keys
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
    allow_mask_recompute: bool = True
    treat_interpolated_as_valid: bool = True

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        emg2pose_group: h5py.Group = self._file[self.HDF5_GROUP]

        # ``timeseries`` is a HDF5 compound Dataset
        self.timeseries: h5py.Dataset = emg2pose_group[self.TIMESERIES]
        assert self.timeseries.dtype.fields is not None
        assert self.EMG in self.timeseries.dtype.fields
        assert self.JOINT_ANGLES in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields

        # Load the metadata entirely into memory as it's rather small
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
        """Load and return a contiguous slice of the timeseries windowed
        by the provided start and end timestamps.

        Args:
            start_t (float): The start time of the window to grab
                (in absolute unix time). Defaults to selecting from the
                beginning of the session. (default: ``-np.inf``).
            end_t (float): The end time of the window to grab
                (in absolute unix time). Defaults to selecting until the
                end of the session. (default: ``np.inf``)
        """
        start_idx, end_idx = self.timestamps.searchsorted([start_t, end_t])
        return self[start_idx:end_idx]

    @property
    def fields(self) -> KeysView[str]:
        """The names of the fields in ``timeseries``."""
        fields: KeysView[str] = self.timeseries.dtype.fields.keys()
        return fields

    @property
    def timestamps(self) -> np.ndarray:
        """EMG timestamps.

        NOTE: This reads the entire sequence of timesetamps from the underlying
        HDF5 file and therefore incurs disk latency. Avoid this in the critical
        path."""
        emg_timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(emg_timestamps) >= 0).all(), "Not monotonic"
        return emg_timestamps

    @property
    def session_name(self) -> str:
        """Unique name of the session."""
        return self.metadata[self.SESSION_NAME]

    @property
    def user(self) -> str:
        """Unique ID of the user this session corresponds to."""
        return self.metadata[self.USER]

    @property
    def no_ik_failure(self):
        if not hasattr(self, "_no_ik_failure"):
            group = self._file[self.HDF5_GROUP]
            mask: np.ndarray | None = None
            if "no_ik_failure" in group:
                mask = group["no_ik_failure"][...]
            elif "ik_failure_mask" in group:
                mask = ~group["ik_failure_mask"][...]
            elif self.allow_mask_recompute:
                joint_angles = self.timeseries[self.JOINT_ANGLES]
                mask = get_ik_failures_mask(joint_angles)
            else:
                raise RuntimeError(
                    "no_ik_failure/ik_failure_mask missing and recompute disabled "
                    f"for session: {self.hdf5_path}"
                )

            mask = np.asarray(mask, dtype=bool)

            # Optionally treat插值样本为有效（默认启用）
            if self.treat_interpolated_as_valid and "interpolated_mask" in group:
                interp = group["interpolated_mask"][...].astype(bool)
                mask = mask | interp

            self._no_ik_failure = mask
        return self._no_ik_failure

    @property
    def interpolated_mask(self) -> np.ndarray:
        if not hasattr(self, "_interpolated_mask"):
            group = self._file[self.HDF5_GROUP]
            if "interpolated_mask" in group:
                mask = group["interpolated_mask"][...]
            else:
                mask = np.zeros(len(self.timeseries), dtype=bool)
            self._interpolated_mask = np.asarray(mask, dtype=bool)
        return self._interpolated_mask

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.session_name} ({len(self)} samples)"


@dataclass
class WindowedEmgDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` corresponding to an instance of
    `Emg2PoseSessionData` that iterates over EMG windows of configurable
    length and stride.

    Args:
        hdf5_path (str): Path to the session file in hdf5 format.
        window_length (int): Size of each window. Specify None for no windowing
            in which case this will be a dataset of length 1 containing the
            entire session. (default: ``None``)
        stride (int): Stride between consecutive windows. Specify None to set
            this to window_length, in which case there will be no overlap
            between consecutive windows. (default: ``window_length``)
        padding (tuple[int, int]): Left and right contextual padding for
            windows in terms of number of raw EMG samples.
        jitter (bool): If True, randomly jitter the offset of each window.
            Use this for training time variability. (default: ``False``)
        transform (Callable): A composed sequence of transforms that takes
            a window/slice of `Emg2PoseSessionData` in the form of a numpy
            structured array and returns a `torch.Tensor` instance.
            (default: ``emg2pose.transforms.ExtractToTensor()``)
        skip_ik_failures (bool): If True, produces only windows with no IK failures.
    """

    hdf5_path: Path
    window_length: InitVar[int | None] = 10_000
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(
        default_factory=transforms.ExtractToTensor
    )
    skip_ik_failures: bool = False
    allow_mask_recompute: bool = True
    treat_interpolated_as_valid: bool = True

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

        # Precompute (start, end) windows based on skip_ik_failures setting
        self.windows: list[tuple[int, int]] = self.precompute_windows()

    def __len__(self) -> int:
        return sum(self._get_block_len(b) for b in self.blocks)

    def _get_block_len(self, block: tuple[float, float]) -> tuple[int]:
        """Get number of samples in a (start, end) time block."""
        return (block[1] - block[0] - self.window_length) // self.stride + 1

    @property
    def session(self):
        if not hasattr(self, "_session"):
            self._session = Emg2PoseSessionData(
                self.hdf5_path,
                allow_mask_recompute=self.allow_mask_recompute,
                treat_interpolated_as_valid=self.treat_interpolated_as_valid,
            )
        return self._session

    @property
    def blocks(self) -> list[tuple[int, int]]:
        """List of (start, end) times to be included in the dataset."""

        if not hasattr(self, "_blocks"):
            
            # Include all time
            if not self.skip_ik_failures:
                if len(self.session) < self.window_length:
                    self._blocks = []
                else:
                    self._blocks = [(0, len(self.session))]

            # Include only time without IK failures
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
        """For each dataset idx, precompute the EMG start and times."""
        windows = []
        cumsum = np.cumsum([0] + [self._get_block_len(b) for b in self.blocks])
        for idx in range(len(self)):
            block_idx = np.searchsorted(cumsum, idx, "right") - 1
            start_idx, end_idx = self.blocks[block_idx]
            relative_idx = idx - cumsum[block_idx]
            windows.append((start_idx + relative_idx * self.stride, end_idx))

        return windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        # Randomly jitter the window offset
        offset, end_idx = self.windows[idx]
        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds, leftover {leftover}")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Expand window to include contextual padding and fetch
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]

        # Extract EMG tensor corresponding to the window
        emg = self.transform(window)
        assert torch.is_tensor(emg)

        # Extract joint angle labels
        joint_angles = window[Emg2PoseSessionData.JOINT_ANGLES]
        joint_angles = torch.as_tensor(joint_angles)

        # Sanitize joint_angles: replace non-finite with 0 and mark invalid in masks
        finite_mask = torch.isfinite(joint_angles).all(dim=1)  # T
        if not finite_mask.all():
            joint_angles = torch.nan_to_num(joint_angles, nan=0.0, posinf=0.0, neginf=0.0)

        # Mask of samples without IK failures
        no_ik_failure = torch.as_tensor(
            self.session.no_ik_failure[window_start:window_end]
        )
        interpolated_mask = torch.as_tensor(
            self.session.interpolated_mask[window_start:window_end]
        )

        # Any non-finite targets are treated as invalid (masked out of loss)
        no_ik_failure = no_ik_failure & finite_mask
        interpolated_mask = interpolated_mask & finite_mask
        return {
            "emg": emg.T,  # CT
            "joint_angles": joint_angles.T,  # CT
            "no_ik_failure": no_ik_failure,  # T
            "interpolated_mask": interpolated_mask,  # T
            "window_start_idx": window_start,
            "window_end_idx": window_end,
        }


class LastStepValidMultiSessionWindowDataset(torch.utils.data.Dataset):
    """Multi-session window dataset that only keeps windows whose *last* target is valid.

    This is intended for "last-step supervision" experiments where we only train on a
    single timestamp per window. It avoids wasting steps on windows where that last
    timestamp is invalid (e.g. IK failure / non-finite).

    Notes:
    - Builds a compact global index: two int32 arrays (session_id, offset).
    - Does not cache full-session masks in RAM; masks are sliced directly from HDF5.
    - Keeps a small LRU of open HDF5 files per worker for throughput.
    """

    def __init__(
        self,
        hdf5_paths: list[Path],
        window_length: int,
        stride: int | None,
        padding: tuple[int, int] = (0, 0),
        jitter: bool = False,
        transform: Transform[np.ndarray, torch.Tensor] = transforms.ExtractToTensor(),
        allow_mask_recompute: bool = False,
        treat_interpolated_as_valid: bool = False,
        require_last_step_valid: bool = True,
        require_finite_last_step: bool = True,
        max_open_sessions: int = 8,
        show_progress: bool = False,
        progress_desc: str = "Indexing windows",
    ) -> None:
        super().__init__()
        self.hdf5_paths = [Path(p) for p in hdf5_paths]
        self.window_length = int(window_length)
        self.stride = int(stride) if stride is not None else int(window_length)
        self.left_padding, self.right_padding = (int(padding[0]), int(padding[1]))
        self.jitter = bool(jitter)
        self.transform = transform
        self.allow_mask_recompute = bool(allow_mask_recompute)
        self.treat_interpolated_as_valid = bool(treat_interpolated_as_valid)
        self.require_last_step_valid = bool(require_last_step_valid)
        self.require_finite_last_step = bool(require_finite_last_step)
        self.max_open_sessions = int(max_open_sessions)
        self.show_progress = bool(show_progress)
        self.progress_desc = str(progress_desc)

        if self.window_length <= 0 or self.stride <= 0:
            raise ValueError("window_length and stride must be positive.")
        if self.left_padding < 0 or self.right_padding < 0:
            raise ValueError("padding values must be non-negative.")
        if self.max_open_sessions <= 0:
            raise ValueError("max_open_sessions must be positive.")

        # Build global index of valid windows: (session_id, offset).
        session_ids: list[np.ndarray] = []
        offsets: list[np.ndarray] = []
        iterator = enumerate(self.hdf5_paths)
        if self.show_progress:
            iterator = tqdm(
                iterator,
                total=len(self.hdf5_paths),
                desc=self.progress_desc,
                unit="session",
            )

        for session_id, path in iterator:
            sid, off = self._build_session_index(session_id=session_id, hdf5_path=path)
            if sid.size == 0:
                continue
            session_ids.append(sid)
            offsets.append(off)

        if session_ids:
            self._session_ids = np.concatenate(session_ids).astype(np.int32, copy=False)
            self._offsets = np.concatenate(offsets).astype(np.int32, copy=False)
        else:
            self._session_ids = np.zeros((0,), dtype=np.int32)
            self._offsets = np.zeros((0,), dtype=np.int32)

        # Per-worker HDF5 LRU cache (initialized lazily in worker process).
        self._open_files: dict[int, h5py.File] = {}
        self._open_timeseries: dict[int, h5py.Dataset] = {}
        self._open_groups: dict[int, h5py.Group] = {}
        self._lru: list[int] = []

    def __len__(self) -> int:
        return int(self._offsets.shape[0])

    def _open_session(self, session_id: int) -> tuple[h5py.File, h5py.Group, h5py.Dataset]:
        if session_id in self._open_files:
            if session_id in self._lru:
                self._lru.remove(session_id)
            self._lru.append(session_id)
            return (
                self._open_files[session_id],
                self._open_groups[session_id],
                self._open_timeseries[session_id],
            )

        while len(self._lru) >= self.max_open_sessions:
            evict = self._lru.pop(0)
            f = self._open_files.pop(evict, None)
            self._open_groups.pop(evict, None)
            self._open_timeseries.pop(evict, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

        path = self.hdf5_paths[session_id]
        f = h5py.File(path, "r")
        group = f[Emg2PoseSessionData.HDF5_GROUP]
        timeseries = group[Emg2PoseSessionData.TIMESERIES]
        self._open_files[session_id] = f
        self._open_groups[session_id] = group
        self._open_timeseries[session_id] = timeseries
        self._lru.append(session_id)
        return f, group, timeseries

    def _build_session_index(self, session_id: int, hdf5_path: Path) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(hdf5_path, "r") as f:
            group = f[Emg2PoseSessionData.HDF5_GROUP]
            timeseries = group[Emg2PoseSessionData.TIMESERIES]
            session_len = int(len(timeseries))

            # `offset` is the start of the *central* window (excluding left padding).
            # We always return fixed-length windows [offset-left_pad : offset+window_len+right_pad].
            min_offset = self.left_padding
            max_offset = session_len - (self.window_length + self.right_padding)
            if max_offset < min_offset:
                return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

            offsets = np.arange(min_offset, max_offset + 1, self.stride, dtype=np.int64)
            if offsets.size == 0:
                return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

            # Last supervised timestep is the last sample of the returned window.
            last_idx = offsets + self.window_length + self.right_padding - 1

            valid = np.ones((offsets.size,), dtype=bool)
            if self.require_last_step_valid:
                no_mask = None
                if "no_ik_failure" in group and len(group["no_ik_failure"]) == session_len:
                    no_mask = group["no_ik_failure"][...].astype(bool, copy=False)
                elif "ik_failure_mask" in group and len(group["ik_failure_mask"]) == session_len:
                    no_mask = ~group["ik_failure_mask"][...].astype(bool, copy=False)
                elif self.allow_mask_recompute:
                    ja = timeseries[Emg2PoseSessionData.JOINT_ANGLES]
                    no_mask = get_ik_failures_mask(ja).astype(bool, copy=False)
                else:
                    return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

                valid = valid & no_mask[last_idx]
                if self.treat_interpolated_as_valid and "interpolated_mask" in group:
                    interp = group["interpolated_mask"][...].astype(bool, copy=False)
                    valid = valid | interp[last_idx]

            if self.require_finite_last_step:
                # Pull joint angles at the last timestep to filter NaN/Inf targets.
                # This is light-weight: only (num_offsets, joint_dim).
                ja_last = timeseries[Emg2PoseSessionData.JOINT_ANGLES][last_idx]
                finite = np.isfinite(ja_last).all(axis=-1)
                valid = valid & finite

            offsets = offsets[valid].astype(np.int32, copy=False)
            session_ids = np.full((offsets.shape[0],), session_id, dtype=np.int32)
            return session_ids, offsets

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        session_id = int(self._session_ids[idx])
        base_offset = int(self._offsets[idx])
        offset = base_offset

        _, group, timeseries = self._open_session(session_id)

        # Jitter within stride for training randomness (kept optional).
        if self.jitter and self.stride > 1:
            session_len = int(len(timeseries))
            max_offset = session_len - (self.window_length + self.right_padding)
            max_jitter = min(self.stride - 1, max(0, max_offset - offset))
            if max_jitter > 0:
                offset = offset + int(np.random.randint(0, max_jitter + 1))

        window_start = offset - self.left_padding
        window_end = offset + self.window_length + self.right_padding
        last_idx = window_end - 1

        # Read only the EMG field for the whole window to keep RAM usage low.
        emg_window = timeseries.fields([Emg2PoseSessionData.EMG])[window_start:window_end]
        emg = self.transform(emg_window)
        assert torch.is_tensor(emg)

        # Read only the last-step joint angles (supervision target).
        ja_last = timeseries.fields([Emg2PoseSessionData.JOINT_ANGLES])[last_idx][
            Emg2PoseSessionData.JOINT_ANGLES
        ]
        ja_last_t = torch.as_tensor(ja_last, dtype=torch.float32)
        finite_last = bool(torch.isfinite(ja_last_t).all().item())
        if not finite_last:
            ja_last_t = torch.nan_to_num(
                ja_last_t, nan=0.0, posinf=0.0, neginf=0.0
            )

        # Slice mask(s) at last_idx directly from HDF5 to avoid caching whole-session masks.
        if "no_ik_failure" in group:
            no_last = bool(group["no_ik_failure"][last_idx])
        elif "ik_failure_mask" in group:
            no_last = not bool(group["ik_failure_mask"][last_idx])
        elif self.allow_mask_recompute:
            no_last = bool(get_ik_failures_mask(np.asarray(ja_last)[None, ...])[0])
        else:
            raise RuntimeError(
                f"Missing no_ik_failure/ik_failure_mask in {self.hdf5_paths[session_id]}"
            )

        if self.treat_interpolated_as_valid and "interpolated_mask" in group:
            no_last = bool(no_last or bool(group["interpolated_mask"][last_idx]))

        # Any non-finite target is treated as invalid (masked out of loss/metrics).
        no_last = bool(no_last and finite_last)

        # If jitter produced an invalid last step (can happen because the index is built
        # on the non-jittered offsets), fall back to the original valid offset.
        if self.jitter and self.require_last_step_valid and not no_last:
            offset = base_offset
            window_start = offset - self.left_padding
            window_end = offset + self.window_length + self.right_padding
            last_idx = window_end - 1

            emg_window = timeseries.fields([Emg2PoseSessionData.EMG])[window_start:window_end]
            emg = self.transform(emg_window)
            assert torch.is_tensor(emg)

            ja_last = timeseries.fields([Emg2PoseSessionData.JOINT_ANGLES])[last_idx][
                Emg2PoseSessionData.JOINT_ANGLES
            ]
            ja_last_t = torch.as_tensor(ja_last, dtype=torch.float32)
            finite_last = bool(torch.isfinite(ja_last_t).all().item())
            if not finite_last:
                ja_last_t = torch.nan_to_num(
                    ja_last_t, nan=0.0, posinf=0.0, neginf=0.0
                )

            if "no_ik_failure" in group:
                no_last = bool(group["no_ik_failure"][last_idx])
            elif "ik_failure_mask" in group:
                no_last = not bool(group["ik_failure_mask"][last_idx])
            elif self.allow_mask_recompute:
                no_last = bool(get_ik_failures_mask(np.asarray(ja_last)[None, ...])[0])
            else:
                raise RuntimeError(
                    f"Missing no_ik_failure/ik_failure_mask in {self.hdf5_paths[session_id]}"
                )
            if self.treat_interpolated_as_valid and "interpolated_mask" in group:
                no_last = bool(no_last or bool(group["interpolated_mask"][last_idx]))
            no_last = bool(no_last and finite_last)

        return {
            "emg": emg.T.contiguous(),  # CT
            "joint_angles": ja_last_t[:, None].contiguous(),  # C1 (last-step only)
            "no_ik_failure": torch.as_tensor([no_last], dtype=torch.bool),  # 1 (last-step only)
            "window_start_idx": window_start,
            "window_end_idx": window_end,
        }
