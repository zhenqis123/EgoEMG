"""Pimforce memmap-based dataset for fast data loading."""

from __future__ import annotations

from egoemg.datasets.angle_utils import pimforce_to_emg2pose_angles

from egoemg.datasets.memmap_windows import MemmapWindowDataset

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from egoemg.augmentation.transforms import Transform
from egoemg.datasets.layout_utils import circular_interpolate, place_sparse_channels


def _convert_emg_layout_for_memmap(
    emg: np.ndarray,
    *,
    channel_alignment_mode: str = "interpolate",
) -> np.ndarray:
    """Convert Pimforce 8 channels to 16 channels via interpolation or sparse placement.

    Applies the corpus channel convention:
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
        anchor_positions = np.arange(emg_t.shape[1], dtype=np.int64) * (
            16 // emg_t.shape[1]
        )
        result = place_sparse_channels(emg_t, 16, anchor_positions)
    else:
        raise ValueError(
            f"channel_alignment_mode must be 'interpolate' or 'sparse', got {channel_alignment_mode}"
        )

    # Transpose back to (16, T)
    return result.T


@dataclass
class PimforceMemmapDataset(MemmapWindowDataset):
    """Windowed dataset for Pimforce memmap format.

    Uses numpy memmap for random access to published corpus arrays.
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        global_start, global_end = self._window_slice(idx)

        emg = np.asarray(
            self._memmaps["emg"][global_start:global_end], dtype=np.float32
        )

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

        result: dict[str, Any] = {
            "emg": emg,
            "label_valid_mask": np.ones(emg.shape[1], dtype=bool),
        }

        if self.return_joint_angles and "joint_angles" in self._memmaps:
            joint_angles = np.asarray(
                self._memmaps["joint_angles"][global_start:global_end], dtype=np.float32
            )
            # Transpose from (T, D) to (D, T) before conversion
            if joint_angles.ndim == 2:
                joint_angles = joint_angles.T
            # Convert to emg2pose ordering (input is already (D, T))
            joint_angles = pimforce_to_emg2pose_angles(
                joint_angles, pose_in_degrees=self.pose_in_degrees
            )
            # Ensure contiguous array for PyTorch collation
            joint_angles = np.ascontiguousarray(joint_angles)
            result["joint_angles"] = joint_angles

        if "force" in self._memmaps:
            force = np.asarray(
                self._memmaps["force"][global_start:global_end], dtype=np.float32
            )
            # Transpose from (T, F) to (F, T) and copy for PyTorch collation
            if force.ndim == 2:
                force = np.ascontiguousarray(force.T)
            else:
                force = np.ascontiguousarray(force)
            result["force"] = force

        if self.transform is not None:
            transformed = self.transform(result)
            if isinstance(transformed, dict):
                result.update(transformed)

        return result
