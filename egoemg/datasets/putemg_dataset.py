from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr

from egoemg.datasets.layout_utils import circular_interpolate


PUTEMG_GESTURE_LABELS: dict[int, str] = {
    -1: "Relax",
    0: "Idle",
    1: "Fist",
    2: "Flexion",
    3: "Extension",
    6: "Pinch thumb-index",
    7: "Pinch thumb-middle",
    8: "Pinch thumb-ring",
    9: "Pinch thumb-small",
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


def _convert_emg_layout(emg: np.ndarray) -> np.ndarray:
    # emg: (T, C) with C=24
    if emg.shape[1] < 8:
        return emg
    choice = np.random.randint(0, 3)
    if choice == 0:
        subset = emg[:, :8]
    elif choice == 1:
        subset = emg[:, 8:16]
    else:
        subset = emg[:, -8:]
    return circular_interpolate(subset, 16)

@dataclass
class PutEmgDataset(Dataset):
    """Windowed dataset for putEMG Zarr store."""

    root_dir: Path
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Any | None = None
    allowed_sessions: Sequence[str] | None = None
    allowed_subjects: Sequence[int] | None = None
    allowed_protocols: Sequence[str] | None = None
    allowed_file_types: Sequence[str] | None = None
    allowed_has_force: bool | None = None
    allowed_has_gesture: bool | None = None

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
        self._session_protocol = _decode_bytes(np.asarray(sessions["protocol"]))
        self._session_file_type = _decode_bytes(np.asarray(sessions["file_type"]))
        self._session_start = np.asarray(sessions["start_idx"], dtype=np.int64)
        self._session_length = np.asarray(sessions["length"], dtype=np.int64)
        self._session_end = np.asarray(sessions["end_idx"], dtype=np.int64)
        self._session_subject = np.asarray(sessions["subject_id"], dtype=np.int32)
        self._session_force_channels = np.asarray(
            sessions["force_channels"], dtype=np.int16
        )
        self._session_traj_channels = np.asarray(
            sessions["traj_channels"], dtype=np.int16
        )
        self._session_has_force = np.asarray(sessions["has_force"], dtype=bool)
        self._session_has_gesture = np.asarray(sessions["has_gesture"], dtype=bool)

        self._emg = root["emg"]
        self._time = root["time"]
        self._force = root["force"]
        self._force_mvc = root["force_mvc"]
        self._traj = root["traj"]
        self._gesture_gt = root["gesture_gt"]
        self._gesture_gt_nf = root["gesture_gt_no_filter"]
        self._video_stamp = root["video_stamp"]
        self._valid = root["valid_mask"]

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

        if self.allowed_subjects:
            allowed = _resolve_int_filter(self._session_subject, self.allowed_subjects)
            mask &= np.isin(self._session_subject, list(allowed))

        if self.allowed_protocols:
            allowed = set(self.allowed_protocols)
            mask &= np.isin(self._session_protocol, list(allowed))

        if self.allowed_file_types:
            allowed = set(self.allowed_file_types)
            mask &= np.isin(self._session_file_type, list(allowed))

        if self.allowed_has_force is not None:
            mask &= self._session_has_force == bool(self.allowed_has_force)

        if self.allowed_has_gesture is not None:
            mask &= self._session_has_gesture == bool(self.allowed_has_gesture)

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
        time = np.asarray(self._time[window_start:window_end], dtype=np.float32)
        force = np.asarray(self._force[window_start:window_end], dtype=np.float32)
        force_mvc = np.asarray(self._force_mvc[window_start:window_end], dtype=np.float32)
        traj = np.asarray(self._traj[window_start:window_end], dtype=np.float32)
        gesture_gt = np.asarray(self._gesture_gt[window_start:window_end], dtype=np.float32)
        gesture_gt_nf = np.asarray(
            self._gesture_gt_nf[window_start:window_end], dtype=np.float32
        )
        video_stamp = np.asarray(
            self._video_stamp[window_start:window_end], dtype=np.float32
        )
        valid = np.asarray(self._valid[window_start:window_end], dtype=bool)

        if self.transform is not None:
            payload = {
                "emg": emg,
                "time": time,
                "force": force,
                "force_mvc": force_mvc,
                "traj": traj,
                "gesture_gt": gesture_gt,
                "gesture_gt_no_filter": gesture_gt_nf,
                "video_stamp": video_stamp,
            }
            transformed = self.transform(payload)
            if isinstance(transformed, dict):
                emg = transformed.get("emg", emg)
                time = transformed.get("time", time)
                force = transformed.get("force", force)
                force_mvc = transformed.get("force_mvc", force_mvc)
                traj = transformed.get("traj", traj)
                gesture_gt = transformed.get("gesture_gt", gesture_gt)
                gesture_gt_nf = transformed.get("gesture_gt_no_filter", gesture_gt_nf)
                video_stamp = transformed.get("video_stamp", video_stamp)
            else:
                emg = transformed

        # Fields:
        # emg: EMG window (24, T), raw sEMG ADC values.
        # time: timestamps (T). Unit:
        # force: force values (10, T) from dynamometer tensometers, sampled at
        #   200 Hz and interpolated to EMG rate.
        # force_mvc: MVC value per frame (T), measured during MVC trajectory.
        # traj: stacked TRAJ_* channels from putEMG (trajectory labels for force
        #   experiment; e.g., TRAJ_1/TRAJ_GT/TRAJ_GT_NO_FILTER in source).
        # gesture_gt: per-frame gesture label (T), DNN-estimated from video.
        # gesture_gt_no_filter: raw video-stream gesture estimation (T).
        #   See PUTEMG_GESTURE_LABELS for label-to-name mapping.
        # video_stamp: timestamps aligned to video stream (T).
        # gesture labels (for TRAJ_*/gesture_*): -1 relax, 0 idle, 1 fist, 2 flexion,
        #   3 extension, 6 pinch thumb-index, 7 pinch thumb-middle, 8 pinch thumb-ring,
        #   9 pinch thumb-small.
        # label_valid_mask: validity mask (T).
        # window_start_idx/window_end_idx: window indices within session.
        # session_idx: integer index into sessions table.
        # session: session identifier string.
        # subject_id: subject identifier.
        # protocol/file_type/has_force/has_gesture: session metadata.
        return {
            "emg": torch.as_tensor(emg).T,
            "time": torch.as_tensor(time),
            "force": torch.as_tensor(force).T,
            "force_mvc": torch.as_tensor(force_mvc),
            "traj": torch.as_tensor(traj).T,
            "gesture_gt": torch.as_tensor(gesture_gt),
            "gesture_gt_no_filter": torch.as_tensor(gesture_gt_nf),
            "video_stamp": torch.as_tensor(video_stamp),
            "label_valid_mask": torch.as_tensor(valid, dtype=torch.bool),
            "window_start_idx": int(window_start_local),
            "window_end_idx": int(window_end_local),
            "session_idx": si,
            "session": self._session_id[si],
            "subject_id": int(self._session_subject[si]),
            "protocol": self._session_protocol[si],
            "file_type": self._session_file_type[si],
            "has_force": bool(self._session_has_force[si]),
            "has_gesture": bool(self._session_has_gesture[si]),
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
        description="Quick smoke test for PutEmgDataset (Zarr)."
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

    dataset = PutEmgDataset(
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
