"""Pimforce memmap-based dataset for fast data loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from egoemg import transforms
from egoemg.transforms import Transform
from egoemg.datasets.layout_utils import circular_interpolate, place_sparse_channels


def _convert_emg_layout_for_memmap(
    emg: np.ndarray,
    *,
    channel_alignment_mode: str = "interpolate",
) -> np.ndarray:
    """Convert Pimforce 8 channels to 16 channels via interpolation or sparse placement.

    Matches the original PimforceDataset implementation:
    - Random flip for data augmentation (when jitter is enabled)
    - Circular interpolation from 8 to 16 channels

    Args:
        emg: EMG data of shape (C, T), where C=8 for Pimforce

    Returns:
        EMG data with 16 channels, shape (16, T)
    """
    # Transpose to (T, C) for interpolation
    if emg.ndim == 2:
        emg_t = emg.T  # (C, T) -> (T, C)
    else:
        emg_t = emg[np.newaxis, :]  # (T,) -> (1, T)

    # Random flip for augmentation (matches original _convert_emg_layout)
    if np.random.rand() < 0.5:
        emg_t = emg_t[:, ::-1]

    if channel_alignment_mode == "interpolate":
        result = circular_interpolate(emg_t, 16)
    elif channel_alignment_mode == "sparse":
        anchor_positions = np.arange(emg_t.shape[1], dtype=np.int64) * (16 // emg_t.shape[1])
        result = place_sparse_channels(emg_t, 16, anchor_positions)
    else:
        raise ValueError(
            f"channel_alignment_mode must be 'interpolate' or 'sparse', got {channel_alignment_mode}"
        )

    # Transpose back to (16, T)
    return result.T


def _pimforce_to_emg2pose_angles(
    pose: np.ndarray, *, pose_in_degrees: bool
) -> np.ndarray:
    """
    Convert Pimforce angle ordering to emg2pose joint ordering.

    PiMforce order: per finger [spread, flex1, flex2, flex3]
    emg2pose order: see egoemg/constants.py
    """
    pose = pose.copy()
    if pose_in_degrees:
        pose[0] -= 20.0
        pose[1] -= 60.0
    else:
        pose[0] -= np.deg2rad(20.0)
        pose[1] -= np.deg2rad(60.0)
    if pose_in_degrees:
        pose = np.deg2rad(pose)

    idx_map = np.array(
        [
            1, 0, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
            12, 13, 14, 15,
            16, 17, 18, 19,
        ],
        dtype=np.int64,
    )
    return pose[idx_map, ...]


@dataclass
class PimforceMemmapDataset(Dataset):
    """Windowed dataset for Pimforce memmap format.

    Uses numpy memmap for zero-copy random access, eliminating zarr overhead.
    """

    root_dir: Path
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Transform[Any, Any] | None = None
    return_joint_angles: bool = True
    pose_in_degrees: bool = True  # Must be True: memmap data is stored in degrees
    channel_alignment_mode: str = "interpolate"

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing memmap directory at {self.root_dir}")

        self.stride = self.stride or self.window_length
        assert self.window_length > 0 and self.stride > 0
        self.left_padding, self.right_padding = self.padding
        assert self.left_padding >= 0 and self.right_padding >= 0
        if self.channel_alignment_mode not in {"interpolate", "sparse"}:
            raise ValueError(
                f"channel_alignment_mode must be 'interpolate' or 'sparse', got {self.channel_alignment_mode}"
            )

        # Load manifest
        manifest_path = self.root_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest at {manifest_path}")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.total_rows = self.manifest["total_rows"]
        self.sessions = self.manifest["sessions"]

        # Build index of windows
        self.file_cumsum = [0]
        session_starts = []
        for sess in self.sessions:
            start = sess["start_idx"]
            length = sess["length"]
            if length < self.window_length:
                continue
            n_windows = (length - self.window_length) // self.stride + 1
            if n_windows <= 0:
                continue
            session_starts.append((start, length, n_windows))
            self.file_cumsum.append(self.file_cumsum[-1] + n_windows)

        self.total_windows = self.file_cumsum[-1]
        self.session_starts = session_starts

        # Load memmaps
        self._memmaps: dict[str, np.memmap] = {}
        self._load_memmaps()

    def _load_memmaps(self) -> None:
        """Load memmap files from manifest."""
        for field_name, field_info in self.manifest["fields"].items():
            filepath = self.root_dir / field_info["filename"]
            dtype = np.dtype(field_info["dtype"])
            shape = tuple(field_info["shape"])
            self._memmaps[field_name] = np.memmap(
                str(filepath), dtype=dtype, mode="r", shape=shape
            )

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Find which session this window belongs to
        sess_idx = int(np.searchsorted(self.file_cumsum, idx, side='right')) - 1
        if sess_idx < 0 or sess_idx >= len(self.session_starts):
            raise IndexError(f"Session index {sess_idx} out of range")

        sess_start, sess_len, _ = self.session_starts[sess_idx]
        base_windows = self.file_cumsum[sess_idx]
        rel_idx = idx - base_windows

        # Calculate window position
        offset = rel_idx * self.stride
        window_end = offset + self.window_length

        # Handle jitter
        leftover = sess_len - window_end
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Read data
        start = max(offset - self.left_padding, 0)
        end = min(offset + self.window_length + self.right_padding, sess_len)

        # Adjust for global offset
        global_start = sess_start + start
        global_end = sess_start + end

        emg = np.asarray(self._memmaps["emg"][global_start:global_end], dtype=np.float32)

        # Transpose from (T, C) to (C, T) to match emg2pose format
        # Copy to ensure writable array for PyTorch collation
        if emg.ndim == 2:
            emg = np.ascontiguousarray(emg.T)
        else:
            emg = np.ascontiguousarray(emg)

        # Convert EMG from 8 to 16 channels via circular interpolation
        emg = _convert_emg_layout_for_memmap(
            emg,
            channel_alignment_mode=self.channel_alignment_mode,
        )

        result: dict[str, Any] = {'emg': emg, 'label_valid_mask': np.ones(emg.shape[1], dtype=bool)}

        if self.return_joint_angles and "joint_angles" in self._memmaps:
            joint_angles = np.asarray(
                self._memmaps["joint_angles"][global_start:global_end], dtype=np.float32
            )
            # Transpose from (T, D) to (D, T) before conversion
            if joint_angles.ndim == 2:
                joint_angles = joint_angles.T
            # Convert to emg2pose ordering (input is already (D, T))
            joint_angles = _pimforce_to_emg2pose_angles(
                joint_angles, pose_in_degrees=self.pose_in_degrees
            )
            # Ensure contiguous array for PyTorch collation
            joint_angles = np.ascontiguousarray(joint_angles)
            result['joint_angles'] = joint_angles

        if "force" in self._memmaps:
            force = np.asarray(
                self._memmaps["force"][global_start:global_end], dtype=np.float32
            )
            # Transpose from (T, F) to (F, T) and copy for PyTorch collation
            if force.ndim == 2:
                force = np.ascontiguousarray(force.T)
            else:
                force = np.ascontiguousarray(force)
            result['force'] = force

        if self.transform is not None:
            transformed = self.transform(result)
            if isinstance(transformed, dict):
                result.update(transformed)

        return result
