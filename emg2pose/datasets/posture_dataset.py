from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr

from emg2pose.datasets.layout_utils import circular_interpolate


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
    return circular_interpolate(data, target_channels)


def _convert_emg_layout(emg: np.ndarray) -> np.ndarray:
    # emg: (T, C)
    if emg.shape[1] < 8:
        return emg
    if np.random.rand() < 0.5:
        subset = emg[:, :8]
        subset = subset[:, [1, 2, 3, 4, 5, 6, 7, 0]]
    else:
        subset = emg[:, -8:]
    return _circular_interpolate(subset, 16)


@dataclass
class PostureDataset(Dataset):
    """Windowed dataset for posture Zarr store."""

    root_dir: Path
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Any | None = None
    allowed_sessions: Sequence[str] | None = None
    allowed_participants: Sequence[int] | None = None
    allowed_days: Sequence[int] | None = None
    allowed_blocks: Sequence[int] | None = None

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
        self._session_participant = np.asarray(sessions["participant_id"], dtype=np.int32)
        self._session_day = np.asarray(sessions["day"], dtype=np.int8)
        self._session_block = np.asarray(sessions["block"], dtype=np.int8)
        self._session_trial = np.asarray(sessions["trial_id"], dtype=np.int16)
        self._session_trial_no = np.asarray(sessions["trial_no"], dtype=np.int16)
        self._session_target = np.asarray(sessions["target_position"], dtype=np.int16)
        self._session_grasp = np.asarray(sessions["grasp"], dtype=np.int16)

        self._emg = root["emg"]
        self._glove = root["glove"]
        self._finger = root["finger"]
        self._time = root["time"]
        self._valid = root["valid_mask"]
        self._target = root["target_position"]
        self._grasp = root["grasp"]

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

        if self.allowed_days:
            allowed = _resolve_int_filter(self._session_day, self.allowed_days)
            mask &= np.isin(self._session_day, list(allowed))

        if self.allowed_blocks:
            allowed = _resolve_int_filter(self._session_block, self.allowed_blocks)
            mask &= np.isin(self._session_block, list(allowed))

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
        emg = _convert_emg_layout(emg)
        glove = np.asarray(self._glove[window_start:window_end], dtype=np.float32)
        finger = np.asarray(self._finger[window_start:window_end], dtype=np.float32)
        time = np.asarray(self._time[window_start:window_end], dtype=np.float32)
        valid = np.asarray(self._valid[window_start:window_end], dtype=bool)

        target = np.asarray(self._target[window_start:window_end], dtype=np.int16)
        grasp = np.asarray(self._grasp[window_start:window_end], dtype=np.int16)

        if self.transform is not None:
            payload = {
                "emg": emg,
                "glove": glove,
                "finger": finger,
                "time": time,
                "target_position": target,
                "grasp": grasp,
            }
            transformed = self.transform(payload)
            if isinstance(transformed, dict):
                emg = transformed.get("emg", emg)
                glove = transformed.get("glove", glove)
                finger = transformed.get("finger", finger)
                time = transformed.get("time", time)
                target = transformed.get("target_position", target)
                grasp = transformed.get("grasp", grasp)
            else:
                emg = transformed

        # Fields:
        # emg: EMG window (16, T).
        # glove: glove sensor readings (18, T).
        # finger: finger sensor readings (5, T).
        # time: timestamps (T). Unit:
        # target_position: target position per frame (T). Values 1-9 for 3x3 grid.
        # grasp: grasp type per frame (T). Values 1-6 for 6 grasp types.
        # label_valid_mask: validity mask (T).
        # window_start_idx/window_end_idx: window indices within session.
        # session_idx: integer index into sessions table.
        # session: session identifier string.
        # participant_id: participant identifier.
        # day: collection day index.
        # block: block index (0-29).
        # trial_id: trial identifier (1-150).
        # trial_no: trial number within block (0-4).
        return {
            "emg": torch.as_tensor(emg).T,
            "glove": torch.as_tensor(glove).T,
            "finger": torch.as_tensor(finger).T,
            "time": torch.as_tensor(time),
            "target_position": torch.as_tensor(target),
            "grasp": torch.as_tensor(grasp),
            "label_valid_mask": torch.as_tensor(valid, dtype=torch.bool),
            "window_start_idx": int(window_start_local),
            "window_end_idx": int(window_end_local),
            "session_idx": si,
            "session": self._session_id[si],
            "participant_id": int(self._session_participant[si]),
            "day": int(self._session_day[si]),
            "block": int(self._session_block[si]),
            "trial_id": int(self._session_trial[si]),
            "trial_no": int(self._session_trial_no[si]),
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
        description="Quick smoke test for PostureDataset (Zarr)."
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

    dataset = PostureDataset(
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
