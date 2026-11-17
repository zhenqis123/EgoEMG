"""Ninapro memmap-based datasets for fast data loading."""

from __future__ import annotations

from egoemg.datasets.memmap_windows import MemmapWindowDataset

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from egoemg.augmentation.transforms import Transform
from egoemg.datasets.layout_utils import circular_interpolate, place_sparse_channels


def _convert_emg_layout_for_memmap(
    emg: np.ndarray,
    db_name: str,
    *,
    channel_alignment_mode: str = "interpolate",
) -> np.ndarray:
    """Convert Ninapro layouts to 16 channels via interpolation or sparse placement.

    Args:
        emg: EMG data of shape (C, T)
        db_name: Database name (DB1, DB2, DB5, etc.)

    Returns:
        EMG data with 16 channels, shape (16, T)
    """
    # Transpose to (T, C) for interpolation
    if emg.ndim == 2:
        emg_t = emg.T  # (C, T) -> (T, C)
    else:
        emg_t = emg[np.newaxis, :]  # (T,) -> (1, T)

    db = db_name.upper().replace("NINAPRO", "").replace("DB", "").strip()
    c = emg_t.shape[1]

    if db in {"1", "2", "3", "4", "6", "7"}:
        # Use first 8 channels, reorder, then interpolate to 16
        subset = emg_t[:, :8]
        subset = subset[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    elif db == "5":
        # DB5 has 16 channels, but we can use either half for augmentation
        if c >= 16:
            subset = emg_t[:, :8]
        else:
            subset = emg_t[:, -8:]
        subset = subset[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    elif db == "8":
        if c >= 16:
            subset = emg_t[:, :8]
        else:
            subset = emg_t[:, -8:]
        subset = subset[:, ::-1]
    else:
        # Unknown DB, return as-is or pad to 16
        if emg_t.shape[1] < 16:
            subset = emg_t
        else:
            result = emg_t[:, :16]
            subset = None

    if "subset" in locals() and subset is not None:
        if channel_alignment_mode == "interpolate":
            result = circular_interpolate(subset, 16)
        elif channel_alignment_mode == "sparse":
            anchor_positions = np.arange(subset.shape[1], dtype=np.int64) * (
                16 // subset.shape[1]
            )
            result = place_sparse_channels(subset, 16, anchor_positions)
        else:
            raise ValueError(
                f"channel_alignment_mode must be 'interpolate' or 'sparse', got {channel_alignment_mode}"
            )

    # Transpose back to (16, T)
    return result.T


@dataclass
class NinaproMemmapDataset(MemmapWindowDataset):
    """Windowed dataset for Ninapro memmap format.

    Uses numpy memmap for random access to published corpus arrays.
    """

    root_dir: Path
    db_name: str = "DB1"
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Transform[Any, Any] | None = None
    return_joint_angles: bool = True
    target_channels: int = 16  # Standardize to 16 channels like emg2pose
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

        # Convert EMG to 16 channels using database-specific layout conversion
        emg = _convert_emg_layout_for_memmap(
            emg,
            self.db_name,
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
            # Ninapro joint_angles are stored in degrees, convert to radians
            joint_angles = np.deg2rad(joint_angles)
            # Transpose from (T, D) to (D, T) and copy for PyTorch collation
            if joint_angles.ndim == 2:
                joint_angles = np.ascontiguousarray(joint_angles.T)
            else:
                joint_angles = np.ascontiguousarray(joint_angles)
            result["joint_angles"] = joint_angles

        if self.transform is not None:
            transformed = self.transform(result)
            if isinstance(transformed, dict):
                result.update(transformed)

        return result


@dataclass
class NinaproDB1MemmapDataset(NinaproMemmapDataset):
    """Ninapro DB1 memmap dataset."""

    db_name: str = "DB1"

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir) / self.db_name
        super().__post_init__()


@dataclass
class NinaproDB2MemmapDataset(NinaproMemmapDataset):
    """Ninapro DB2 memmap dataset."""

    db_name: str = "DB2"

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir) / self.db_name
        super().__post_init__()


@dataclass
class NinaproDB5MemmapDataset(NinaproMemmapDataset):
    """Ninapro DB5 memmap dataset."""

    db_name: str = "DB5"

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir) / self.db_name
        super().__post_init__()
