from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import zarr

from emg2pose.datasets.layout_utils import circular_interpolate

GRABMYO_GESTURE_LABELS: dict[int, str] = {
    1: "Lateral prehension",
    2: "Thumb adduction",
    3: "Thumb and little finger opposition",
    4: "Thumb and index finger opposition",
    5: "Thumb and index finger extension",
    6: "Thumb and little finger extension",
    7: "Index and middle finger extension",
    8: "Little finger extension",
    9: "Index finger extension",
    10: "Thumb finger extension",
    11: "Wrist extension",
    12: "Wrist flexion",
    13: "Forearm supination",
    14: "Forearm pronation",
    15: "Hand open",
    16: "Hand close",
    17: "Rest",
}


def _decode_bytes(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            decoded.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(v))
    return decoded


def _resolve_int_filter(values: np.ndarray, allowed: Sequence[int] | None) -> set[int]:
    if not allowed:
        return set()
    return {int(v) for v in allowed}

def _circular_interpolate(data: np.ndarray, target_channels: int) -> np.ndarray:
    """环形布局线性插值（预计算矩阵，直接矩阵乘法）。"""
    return circular_interpolate(data, target_channels)

@dataclass
class GrabMyoDataset(Dataset):
    """Windowed dataset for GrabMyo Zarr store."""

    root_dir: Path
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Any | None = None
    allowed_sessions: Sequence[str] | None = None
    allowed_participants: Sequence[int] | None = None
    allowed_session_numbers: Sequence[int] | None = None
    allowed_gestures: Sequence[int] | None = None

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing Zarr store at {self.root_dir}")

        self.stride = self.stride or self.window_length
        assert self.window_length > 0 and self.stride > 0

        self.left_padding, self.right_padding = self.padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        self._root = zarr.open_group(str(self.root_dir), mode="r")
        self._load_catalog()
        self._build_blocks_index()

    def _load_catalog(self) -> None:
        root = self._root
        sessions = root["sessions"]
        self._session_id = _decode_bytes(np.asarray(sessions["session_id"]))
        self._session_filename = _decode_bytes(np.asarray(sessions["filename"]))
        self._session_start = np.asarray(sessions["start_idx"], dtype=np.int64)
        self._session_length = np.asarray(sessions["length"], dtype=np.int64)
        self._session_end = np.asarray(sessions["end_idx"], dtype=np.int64)
        self._session_participant = np.asarray(sessions["participant_id"], dtype=np.int16)
        self._session_number = np.asarray(sessions["session_number"], dtype=np.int16)
        self._session_repetition = np.asarray(sessions["repetition"], dtype=np.int8)
        self._session_gesture = np.asarray(sessions["gesture_id"], dtype=np.int16)

        self._emg = root["emg"]
        self._time = root["time"]
        self._gesture_id = root["gesture_id"]
        self._valid = root["valid_mask"]
        self._gesture_labels = (
            _decode_bytes(np.asarray(root["gesture_labels"]))
            if "gesture_labels" in root
            else None
        )

    def _filter_session_indices(self) -> np.ndarray:
        n_sessions = len(self._session_id)
        mask = np.ones((n_sessions,), dtype=bool)

        if self.allowed_sessions:
            allowed = set(self.allowed_sessions)
            mask &= np.array(
                [
                    (sid in allowed) or (fn in allowed)
                    for sid, fn in zip(self._session_id, self._session_filename)
                ],
                dtype=bool,
            )

        if self.allowed_participants:
            allowed = _resolve_int_filter(self._session_participant, self.allowed_participants)
            mask &= np.isin(self._session_participant, list(allowed))

        if self.allowed_session_numbers:
            allowed = _resolve_int_filter(self._session_number, self.allowed_session_numbers)
            mask &= np.isin(self._session_number, list(allowed))

        if self.allowed_gestures:
            allowed = _resolve_int_filter(self._session_gesture, self.allowed_gestures)
            mask &= np.isin(self._session_gesture, list(allowed))

        return np.nonzero(mask)[0].astype(np.int64)

    def _build_blocks_index(self) -> None:
        allowed_sessions = self._filter_session_indices()
        block_session_idx: list[int] = []
        block_start: list[int] = []
        block_end: list[int] = []
        block_lengths: list[int] = []

        for si in allowed_sessions:
            slen = int(self._session_length[si])
            if slen < self.window_length:
                continue
            n = (slen - self.window_length) // self.stride + 1
            block_session_idx.append(int(si))
            block_start.append(0)
            block_end.append(slen)
            block_lengths.append(n)

        self._block_session_idx = np.asarray(block_session_idx, dtype=np.int32)
        self._block_start = np.asarray(block_start, dtype=np.int64)
        self._block_end = np.asarray(block_end, dtype=np.int64)
        self._block_cumsum = np.cumsum(np.asarray([0] + block_lengths, dtype=np.int64))

    def __len__(self) -> int:
        return int(self._block_cumsum[-1])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        bi = int(np.searchsorted(self._block_cumsum, idx, side="right") - 1)
        si = int(self._block_session_idx[bi])
        start_idx = int(self._block_start[bi])
        end_idx = int(self._block_end[bi])
        rel = int(idx - self._block_cumsum[bi])

        offset = start_idx + rel * self.stride
        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        session_start = int(self._session_start[si])
        session_end = int(self._session_end[si])

        window_start = max(session_start + offset - self.left_padding, session_start)
        window_end = min(
            session_start + offset + self.window_length + self.right_padding,
            session_end,
        )
        window_start_local = window_start - session_start
        window_end_local = window_end - session_start

        emg = np.asarray(self._emg[window_start:window_end], dtype=np.float32)
        time = np.asarray(self._time[window_start:window_end], dtype=np.float32)
        gesture_id = np.asarray(self._gesture_id[window_start:window_end], dtype=np.int16)
        valid = np.asarray(self._valid[window_start:window_end], dtype=bool)
        
        emg = self._convert_layout(emg)  # Apply layout conversion
        
        if self.transform is not None:
            payload = {
                "emg": emg,
                "time": time,
                "gesture_id": gesture_id,
            }
            transformed = self.transform(payload)
            if isinstance(transformed, dict):
                emg = transformed.get("emg", emg)
                time = transformed.get("time", time)
                gesture_id = transformed.get("gesture_id", gesture_id)
            else:
                emg = transformed

        # Fields:
        # emg: EMG window (28, T). Forearm 16 + wrist 12 channels at 2048 Hz.
        # time: timestamps (T). Unit:
        # gesture_id: per-frame gesture label (T), 17 gestures in GrabMyo.
        #   See GRABMYO_GESTURE_LABELS for the semantic mapping.
        # label_valid_mask: validity mask (T).
        # window_start_idx/window_end_idx: window indices within session.
        # session_idx: integer index into sessions table.
        # session: session identifier string.
        # participant_id/session_number/repetition: session metadata. Meaning:
        # gesture_labels: (optional) list of label names for gesture_id indices.
        sample: dict[str, Any] = {
            "emg": torch.as_tensor(emg).T,
            "time": torch.as_tensor(time),
            "gesture_id": torch.as_tensor(gesture_id),
            "label_valid_mask": torch.as_tensor(valid, dtype=torch.bool),
            "window_start_idx": int(window_start_local),
            "window_end_idx": int(window_end_local),
            "session_idx": si,
            "session": self._session_id[si],
            "participant_id": int(self._session_participant[si]),
            "session_number": int(self._session_number[si]),
            "repetition": int(self._session_repetition[si]),
        }
        if self._gesture_labels is not None:
            sample["gesture_labels"] = list(self._gesture_labels)
        return sample

    def _convert_layout(self, emg: np.ndarray) -> np.ndarray:
            '''Four available different layouts conversion with Spatial Interpolation'''
            # Random select a conversion layout
            layout_type = random.randint(0, 3)
            
            # 定义目标通道数
            TARGET_CH = 16
            
            if layout_type == 0:
                # --- 8 Channels Layout A ---
                emg_subset = emg[:, :8]
                emg_subset = emg_subset[:, ::-1]          # Reverse
                emg_subset = np.roll(emg_subset, 2, axis=1) # Shift (90度)
                
                # 使用插值代替 Padding 0
                # 8->16 插值会自动在每两个数中间填入平均值，比填0更平滑
                return _circular_interpolate(emg_subset, TARGET_CH)

            elif layout_type == 1:
                # --- 8 Channels Layout B ---
                emg_subset = emg[:, 8:16]
                emg_subset = emg_subset[:, ::-1]          # Reverse
                emg_subset = np.roll(emg_subset, 2, axis=1) # Shift (90度)
                
                return _circular_interpolate(emg_subset, TARGET_CH)

            elif layout_type == 2:
                # --- 6 Channels Layout A ---
                emg_subset = emg[:, 16:22]
                emg_subset = emg_subset[:, ::-1]          # Reverse
                emg_subset = np.roll(emg_subset, 1, axis=1) # Shift (60度)
                
                # 6->16 必须使用插值，单纯 Padding 无法处理非整数倍扩展
                return _circular_interpolate(emg_subset, TARGET_CH)

            elif layout_type == 3:
                # --- 6 Channels Layout B ---
                emg_subset = emg[:, 22:28]
                emg_subset = emg_subset[:, ::-1]          # Reverse
                emg_subset = np.roll(emg_subset, 1, axis=1) # Shift (60度)
                
                return _circular_interpolate(emg_subset, TARGET_CH)
            
            return None # Should not reach here

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
        description="Quick smoke test for GrabMyoDataset (Zarr)."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Path to the Zarr dataset root.",
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
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    dataset = GrabMyoDataset(
        root_dir=args.root_dir,
        window_length=args.window_length,
        stride=args.stride,
        padding=args.padding,
        jitter=args.jitter,
    )

    print(f"Sessions: {len(dataset._session_id)}")
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
