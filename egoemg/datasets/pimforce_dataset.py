from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr

from egoemg.datasets.layout_utils import circular_interpolate


def _decode_bytes(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            decoded.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(v))
    return decoded


def _circular_interpolate(data: np.ndarray, target_channels: int) -> np.ndarray:
    return circular_interpolate(data, target_channels)


def _convert_emg_layout(emg: np.ndarray) -> np.ndarray:
    """Random flip then interpolate to 16 channels."""
    # emg: (C, T)
    if np.random.rand() < 0.5:
        emg = emg[::-1, :]
    emg_tc = emg.T  # (T, C)
    emg_tc = _circular_interpolate(emg_tc, 16)
    return emg_tc.T


def _pimforce_to_emg2pose_angles(
    pose: np.ndarray, *, pose_in_degrees: bool
) -> np.ndarray:
    """
    Convert PiMforce angle ordering to emg2pose joint ordering.

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


class _ZarrPimforceStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.root = zarr.open_group(str(self.root_dir), mode="r")
        self.emg = self.root["emg"]
        self.joint = self.root["joint_angles"]
        self.force = self.root["force"]
        self.valid = self.root["valid_mask"]
        sessions = self.root["sessions"]
        self.session_id = np.asarray(sessions["session_id"], dtype=np.int32)
        self.user_id = np.asarray(sessions["user_id"], dtype=np.int32)
        self.file_id = np.asarray(sessions["file_id"], dtype=np.int32)
        self.session_name = _decode_bytes(np.asarray(sessions["session_name"]))
        self.filename = _decode_bytes(np.asarray(sessions["filename"]))
        self.start_idx = np.asarray(sessions["start_idx"], dtype=np.int64)
        self.length = np.asarray(sessions["length"], dtype=np.int64)
        self.end_idx = np.asarray(sessions["end_idx"], dtype=np.int64)
        self.original_length = np.asarray(sessions["original_length"], dtype=np.int64)
        self.duration = np.asarray(sessions["duration"], dtype=np.float32)


@dataclass
class PimforceDataset(Dataset):
    """Windowed dataset for PiMForce Zarr store."""

    root_dirs: list[Path]
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    pose_in_degrees: bool = True
    transform: Callable[[dict[str, np.ndarray]], Any] | None = None
    allowed_sessions: Sequence[int] | None = None
    allowed_users: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if not self.root_dirs:
            raise ValueError("root_dirs must contain at least one Zarr root")

        self.store = _ZarrPimforceStore(self.root_dirs[0])

        # Set default stride
        self.stride = self.stride or self.window_length
        if self.window_length <= 0 or self.stride <= 0:
            raise ValueError("window_length and stride must be positive")

        self.left_padding, self.right_padding = self.padding
        if self.left_padding < 0 or self.right_padding < 0:
            raise ValueError("padding values must be non-negative")

        self._build_blocks_index()

    def _build_blocks_index(self) -> None:
        """Build sliding window index across all sessions."""
        session_ids = self.store.session_id
        user_ids = self.store.user_id
        indices = np.arange(len(session_ids), dtype=np.int64)

        # Apply filters
        if self.allowed_sessions is not None:
            allowed = np.asarray(self.allowed_sessions, dtype=np.int64)
            indices = indices[np.isin(session_ids, allowed)]

        if self.allowed_users is not None:
            allowed = np.asarray(self.allowed_users, dtype=np.int64)
            indices = indices[np.isin(user_ids, allowed)]

        # Calculate number of windows per session
        block_session_idx: list[int] = []
        block_start: list[int] = []
        block_end: list[int] = []
        block_lengths: list[int] = []

        for si in indices:
            slen = int(self.store.length[si])
            if slen < self.window_length:
                continue

            # Core formula: number of windows in this session
            n = (slen - self.window_length) // self.stride + 1

            block_session_idx.append(int(si))
            block_start.append(0)  # relative start within session
            block_end.append(slen)
            block_lengths.append(n)

        # Build cumulative sum array for fast indexing
        self._block_session_idx = np.asarray(block_session_idx, dtype=np.int32)
        self._block_start = np.asarray(block_start, dtype=np.int64)
        self._block_end = np.asarray(block_end, dtype=np.int64)
        self._block_cumsum = np.cumsum(
            np.asarray([0] + block_lengths, dtype=np.int64)
        )

    def __len__(self) -> int:
        """Return total number of windows across all sessions."""
        return int(self._block_cumsum[-1])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Binary search to locate session
        bi = int(np.searchsorted(self._block_cumsum, idx, side="right") - 1)
        si = int(self._block_session_idx[bi])
        start_idx = int(self._block_start[bi])
        end_idx = int(self._block_end[bi])

        # Calculate window offset within session
        rel = int(idx - self._block_cumsum[bi])
        offset = start_idx + rel * self.stride

        # Apply jitter if enabled
        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Get session's global range
        session_start = int(self.store.start_idx[si])
        session_end = int(self.store.end_idx[si])

        # Calculate window's global range (with padding)
        window_start = max(session_start + offset - self.left_padding, session_start)
        window_end = min(
            session_start + offset + self.window_length + self.right_padding,
            session_end,
        )
        window_start_local = window_start - session_start

        # Read data from Zarr
        emg = np.asarray(
            self.store.emg[window_start:window_end],
            dtype=np.float32
        ).T  # (C, T)

        pose = np.asarray(
            self.store.joint[window_start:window_end],
            dtype=np.float32
        ).T  # (C, T)

        # EMG channel conversion (8 → 16)
        emg = _convert_emg_layout(emg)

        # Angle conversion
        joint_angles = _pimforce_to_emg2pose_angles(
            pose, pose_in_degrees=self.pose_in_degrees
        )

        # Apply transform
        if self.transform is not None:
            transformed = self.transform({"emg": emg, "joint_angles": joint_angles})
            if isinstance(transformed, dict):
                emg = transformed.get("emg", emg)
                joint_angles = transformed.get("joint_angles", joint_angles)
            elif isinstance(transformed, tuple) and len(transformed) == 2:
                emg, joint_angles = transformed
            else:
                emg = transformed

        # Build return dictionary
        emg_t = torch.as_tensor(emg)
        joint_angles_t = torch.as_tensor(joint_angles)
        time_len = emg_t.shape[-1]

        # Create validity mask
        valid_len = int(self.store.original_length[si])
        mask = torch.zeros(time_len, dtype=torch.bool)
        valid_end = min(offset + self.window_length, valid_len)
        if valid_end > offset:
            mask_start = max(0, offset - (window_start_local - start_idx))
            mask_end = mask_start + (valid_end - offset)
            mask[mask_start:mask_end] = True

        # Fields:
        # emg: EMG window (16, T). Forearm sEMG channels from PiMForce.
        # joint_angles: converted PiMForce 3D hand posture (20, T). Unit: radians.
        # label_valid_mask: validity mask (T) based on available length.
        # window_start_idx/window_end_idx: window indices within session.
        # session_idx: integer index into sessions table.
        # user_id: integer subject identifier.
        # session_id: integer session identifier.
        return {
            "emg": emg_t,
            "joint_angles": joint_angles_t,
            "label_valid_mask": mask,
            "window_start_idx": offset,
            "window_end_idx": offset + self.window_length,
            "session_idx": si,
            "user_id": int(self.store.user_id[si]),
            "session_id": int(self.store.session_id[si]),
        }


def _describe_value(value: Any) -> str:
    if torch.is_tensor(value):
        return f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    return f"{type(value).__name__}"


def _print_sample(sample: dict[str, Any]) -> None:
    for key in sorted(sample.keys()):
        print(f"  {key}: {_describe_value(sample[key])}")


def _parse_padding(text: str) -> tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("padding must be 'left,right'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick smoke test for PimforceDataset (Zarr)."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Path to the Pimforce Zarr root.",
    )
    parser.add_argument("--window-length", type=int, default=10_000)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--padding",
        type=_parse_padding,
        default=(0, 0),
        help="Left,right padding (e.g., 0,0).",
    )
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--pose-in-degrees", action="store_true")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    dataset = PimforceDataset(
        root_dirs=[args.root_dir],
        window_length=args.window_length,
        stride=args.stride,
        padding=args.padding,
        jitter=args.jitter,
        pose_in_degrees=args.pose_in_degrees,
    )

    print(f"Sessions: {len(dataset.store.session_id)}")
    print(f"Total windows: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Dataset is empty with current filters.")
        return

    n = min(args.num_samples, len(dataset))
    if args.sequential:
        indices = list(range(n))
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.integers(0, len(dataset), size=n).tolist()

    for i, idx in enumerate(indices):
        print(f"Sample {i} (idx={idx}):")
        sample = dataset[int(idx)]
        _print_sample(sample)


if __name__ == "__main__":
    main()
