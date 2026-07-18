from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr

from emg2pose.datasets.layout_utils import circular_interpolate


NINAPRO_REST_LABEL = "Rest"
NINAPRO_EXERCISE_LABELS: dict[str, list[str]] = {
    "A": [
        "Index flexion",
        "Index extension",
        "Middle flexion",
        "Middle extension",
        "Ring flexion",
        "Ring extension",
        "Little finger flexion",
        "Little finger extension",
        "Thumb adduction",
        "Thumb abduction",
        "Thumb flexion",
        "Thumb extension",
    ],
    "B": [
        "Thumb up",
        "Extension of index and middle, flexion of the others",
        "Flexion of ring and little finger, extension of the others",
        "Thumb opposing base of little finger",
        "Abduction all fingers",
        "Fingers flexed together in fist",
        "Pointing index",
        "Adduction of extended fingers",
        "Wrist supination (axis: middle finger)",
        "Wrist pronation (axis: middle finger)",
        "Wrist supination (axis: little finger)",
        "Wrist pronation (axis: little finger)",
        "Wrist flexion",
        "Wrist extension",
        "Wrist radial deviation",
        "Wrist ulnar deviation",
        "Wrist extension with closed hand",
    ],
    "C": [
        "Large diameter grasp",
        "Small diameter grasp (power grip)",
        "Fixed hook grasp",
        "Index finger extension grasp",
        "Medium wrap",
        "Ring grasp",
        "Prismatic four fingers grasp",
        "Stick grasp",
        "Writing tripod grasp",
        "Power sphere grasp",
        "Three finger sphere grasp",
        "Precision sphere grasp",
        "Tripod grasp",
        "Prismatic pinch grasp",
        "Tip pinch grasp",
        "Quadpod grasp",
        "Lateral grasp",
        "Parallel extension grasp",
        "Extension type grasp",
        "Power disk grasp",
        "Open a bottle with a tripod grasp",
        "Turn a screw (grasp the screwdriver with a stick grasp)",
        "Cut something (grasp the knife with an index finger extension grasp)",
    ],
    "D": [
        "Flexion of the little finger",
        "Flexion of the ring finger",
        "Flexion of the middle finger",
        "Flexion of the index finger",
        "Abduction of the thumb",
        "Flexion of the thumb",
        "Flexion of index and little finger",
        "Flexion of ring and middle finger",
        "Flexion of index finger and thumb",
    ],
}
NINAPRO_DEFAULT_EXERCISE_ORDER = ("A", "B", "C")

def _circular_interpolate(data: np.ndarray, target_channels: int) -> np.ndarray:
    return circular_interpolate(data, target_channels)


def _convert_emg_layout(emg: np.ndarray, db_name: str) -> np.ndarray:
    """Convert Ninapro layouts to 16 channels via interpolation."""
    db = db_name.upper().replace("NINAPRO", "").replace("DB", "").strip()
    c = emg.shape[1]
    if db in {"1", "2", "3", "4", "6", "7"}:
        subset = emg[:, :8]
        subset = subset[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        return _circular_interpolate(subset, 16)
    if db == "5":
        if c >= 16 and np.random.rand() < 0.5:
            subset = emg[:, :8]
        else:
            subset = emg[:, -8:]
        subset = subset[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        return _circular_interpolate(subset, 16)
    if db == "8":
        if c >= 16 and np.random.rand() < 0.5:
            subset = emg[:, :8]
        else:
            subset = emg[:, -8:]
        subset = subset[:, ::-1]
        return _circular_interpolate(subset, 16)
    return emg


def build_ninapro_label_map(
    exercises: Sequence[str] = NINAPRO_DEFAULT_EXERCISE_ORDER,
    *,
    include_rest: bool = True,
) -> dict[int, str]:
    labels: list[str] = []
    if include_rest:
        labels.append(NINAPRO_REST_LABEL)
    for exercise in exercises:
        labels.extend(NINAPRO_EXERCISE_LABELS.get(exercise, []))
    return {idx: name for idx, name in enumerate(labels)}


def _decode_bytes(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            decoded.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(v))
    return decoded


@dataclass
class NinaproDataset(Dataset):
    """Windowed dataset for Ninapro relabeled Zarr stores (per-DB)."""

    db_name: ClassVar[str] = ""
    root_dir: Path | None = None
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    return_joint_angles: bool = True
    return_gesture_labels: bool = False
    use_gesture_as_joint_angles: bool = True
    transform: Any | None = None

    def __post_init__(self) -> None:
        if self.root_dir is None:
            self.root_dir = Path("data/emg_corpus/Ninapro_relabeled_zarr") / self.db_name
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
        self._session_user = np.asarray(sessions["user"], dtype=np.int32)
        self._session_exercise = np.asarray(sessions["exercise"], dtype=np.int16)
        self._session_dataset = _decode_bytes(np.asarray(sessions["dataset"]))
        self._session_start = np.asarray(sessions["start_idx"], dtype=np.int64)
        self._session_length = np.asarray(sessions["length"], dtype=np.int64)
        self._session_end = np.asarray(sessions["end_idx"], dtype=np.int64)

        self._emg = root["emg"]
        self._gesture_id = root["gesture_id"]
        self._valid = root["valid_mask"]
        self._joint = root["joint_angles"] if "joint_angles" in root else None
        self._gesture_labels = (
            _decode_bytes(np.asarray(root["gesture_labels"]))
            if "gesture_labels" in root
            else None
        )

    def _build_blocks_index(self) -> None:
        block_session_idx: list[int] = []
        block_start: list[int] = []
        block_end: list[int] = []
        block_lengths: list[int] = []

        for si in range(len(self._session_id)):
            slen = int(self._session_length[si])
            if slen < self.window_length:
                continue
            n = (slen - self.window_length) // self.stride + 1
            if n <= 0:
                continue
            block_session_idx.append(si)
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
        emg = _convert_emg_layout(emg, self.db_name)
        gesture_id = np.asarray(
            self._gesture_id[window_start:window_end], dtype=np.int32
        )

        joint_angles = None
        if self.return_joint_angles and self._joint is not None:
            joint_angles = np.asarray(self._joint[window_start:window_end], dtype=np.float32)
            joint_angles = np.deg2rad(joint_angles)
        elif self.return_joint_angles and self.use_gesture_as_joint_angles:
            joint_angles = gesture_id.astype(np.float32, copy=False)[:, None]

        if self.transform is not None:
            payload: Any
            if joint_angles is None:
                payload = {"emg": emg}
            else:
                payload = {"emg": emg, "joint_angles": joint_angles}
            transformed = self.transform(payload)
            if isinstance(transformed, dict):
                emg = transformed.get("emg", emg)
                joint_angles = transformed.get("joint_angles", joint_angles)
            elif isinstance(transformed, tuple) and len(transformed) == 2:
                emg, joint_angles = transformed
            else:
                emg = transformed

        length = min(emg.shape[0], gesture_id.shape[0])
        if joint_angles is not None:
            length = min(length, joint_angles.shape[0])
            joint_angles = joint_angles[:length]
        emg = emg[:length]
        gesture_id = gesture_id[:length]

        def _finite_mask(arr: Any) -> np.ndarray:
            """Return True for timesteps where at least one channel is finite.

            Per-channel NaN handling is delegated to PretrainWrapperDataset.
            """
            if torch.is_tensor(arr):
                return torch.isfinite(arr).any(dim=1).cpu().numpy()
            return np.isfinite(arr).any(axis=1)

        label_valid_mask = np.ones(length, dtype=bool)
        if joint_angles is not None and getattr(joint_angles, "ndim", 0) == 2:
            finite_mask = _finite_mask(joint_angles)
            if not finite_mask.all():
                if torch.is_tensor(joint_angles):
                    joint_angles = torch.nan_to_num(
                        joint_angles, nan=0.0, posinf=0.0, neginf=0.0
                    )
                else:
                    joint_angles = np.nan_to_num(
                        joint_angles, nan=0.0, posinf=0.0, neginf=0.0
                    )
            label_valid_mask &= finite_mask

        # Fields:
        # emg: EMG window (10/12/16, T), float32. Channel counts by DB:
        #   DB1=10, DB2=12, DB3=12, DB4=12, DB5=16, DB6=16, DB7=12, DB8=16.
        # joint_angles: joint angles (22, T) for DB1/DB2/DB5 (kinematic data
        #   available in these DBs). Otherwise None.
        #   When use_gesture_as_joint_angles=True and joint_angles is missing,
        #   joint_angles becomes (1, T) from gesture_id.
        # gesture_id: per-frame gesture class id (T). Label counts reported in
        #   Ninapro docs/papers: DB1=52, DB2=50, DB3=50, DB4=53, DB5=53,
        #   DB6=7, DB7=41 (DB8 not reported there).
        #   DB1/DB4/DB5 include a rest position in addition to movements.
        #   See NINAPRO_EXERCISE_LABELS / build_ninapro_label_map for a label map.
        # label_valid_mask: validity mask (T) from finite joint angles.
        # window_start_idx/window_end_idx: window indices within session.
        # session_idx: integer index into sessions table.
        # session: session identifier string.
        # user: subject identifier string (from session metadata).
        # side: recording side (not provided in dataset).
        # gesture_labels: (optional) list of label names for gesture_id indices.
        sample: dict[str, Any] = {
            "emg": torch.as_tensor(emg).T,
            "joint_angles": torch.as_tensor(joint_angles).T if joint_angles is not None else None,
            "gesture_id": torch.as_tensor(gesture_id),
            "label_valid_mask": torch.as_tensor(label_valid_mask, dtype=torch.bool),
            "window_start_idx": int(window_start_local),
            "window_end_idx": int(window_end_local),
            "session_idx": si,
            "session": self._session_id[si],
            "user": str(self._session_user[si]),
            "side": "unknown",
        }
        if self.return_gesture_labels and self._gesture_labels is not None:
            sample["gesture_labels"] = list(self._gesture_labels)
        return sample


class NinaproDB1Dataset(NinaproDataset):
    db_name = "DB1"


class NinaproDB2Dataset(NinaproDataset):
    db_name = "DB2"


class NinaproDB3Dataset(NinaproDataset):
    db_name = "DB3"


class NinaproDB4Dataset(NinaproDataset):
    db_name = "DB4"


class NinaproDB5Dataset(NinaproDataset):
    db_name = "DB5"


class NinaproDB6Dataset(NinaproDataset):
    db_name = "DB6"


class NinaproDB7Dataset(NinaproDataset):
    db_name = "DB7"


class NinaproDB8Dataset(NinaproDataset):
    db_name = "DB8"


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


def _resolve_dataset_class(db: str | None) -> type[NinaproDataset]:
    if db is None:
        return NinaproDataset
    db = db.upper().replace("NINAPRO", "").replace("DB", "").strip()
    mapping: dict[str, type[NinaproDataset]] = {
        "1": NinaproDB1Dataset,
        "2": NinaproDB2Dataset,
        "3": NinaproDB3Dataset,
        "4": NinaproDB4Dataset,
        "5": NinaproDB5Dataset,
        "6": NinaproDB6Dataset,
        "7": NinaproDB7Dataset,
        "8": NinaproDB8Dataset,
    }
    if db not in mapping:
        raise ValueError(f"Unknown DB: {db}")
    return mapping[db]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick smoke test for NinaproDataset (Zarr)."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Path to DB Zarr root (if not using --db).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="DB number to use (e.g., 1, DB1).",
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
    parser.add_argument("--return-joint-angles", action="store_true")
    parser.add_argument("--return-gesture-labels", action="store_true")
    parser.add_argument("--use-gesture-as-joint-angles", action="store_true")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    cls = _resolve_dataset_class(args.db)
    if cls is NinaproDataset and args.root_dir is None:
        raise ValueError("--root-dir is required when --db is not provided.")

    dataset = cls(
        root_dir=args.root_dir,
        window_length=args.window_length,
        stride=args.stride,
        padding=args.padding,
        jitter=args.jitter,
        return_joint_angles=args.return_joint_angles,
        return_gesture_labels=args.return_gesture_labels,
        use_gesture_as_joint_angles=args.use_gesture_as_joint_angles,
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
