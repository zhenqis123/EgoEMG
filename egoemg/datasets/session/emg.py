"""EMGFormer reader for the independent-clock EgoEMG session-v2 archive."""

from __future__ import annotations

from egoemg.datasets.session.schema import SPLIT_IDS, _read, _array, _valid_label_key

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from egoemg.datasets.evaluation_partition import load_evaluation_partition

REQUIRED = (
    "emg_filtered",
    "joint_angles",
    "wrist_angles",
    "wrist_valid",
    "split_id",
    "timestamps",
)


@dataclass(frozen=True)
class _SessionHand:
    session_id: str
    source_group: str
    subject: str
    folder: Path
    arrays: dict[str, dict[str, Any]]
    rows: int


class EgoEmgSessionDataset(Dataset):
    """Return 8-channel EMG and 22-channel angles from active session arrays.

    Windows stay inside one split, physical clock segment, source recording,
    and contiguous valid EMG span. Finger supervision uses the published label
    validity; wrist supervision additionally uses the per-row wrist mask.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        target_hand: str,
        allowed_splits: Sequence[str],
        window_length: int,
        stride: int,
        emg_field: str = "emg_filtered",
        norm_mode: str | None = None,
        norm_stats_path: str | Path | None = None,
        transform: Any = None,
        dataset_name: str = "egoemg_session_v2",
        max_open_sessions: int = 4,
        jitter: bool = False,
        session_id_prefix: str | None = None,
        evaluation_partition_path: str | Path | None = None,
        evaluation_role: str | None = None,
    ) -> None:
        if target_hand not in {"left", "right"}:
            raise ValueError(f"Unknown hand: {target_hand}")
        if emg_field not in {"emg_raw", "emg_filtered"}:
            raise ValueError(f"Unknown EMG field: {emg_field}")
        if window_length <= 0 or stride <= 0:
            raise ValueError("window_length and stride must be positive")
        if not allowed_splits or any(name not in SPLIT_IDS for name in allowed_splits):
            raise ValueError(f"Unknown or empty split list: {allowed_splits}")
        self.dataset_dir = Path(dataset_dir).resolve()
        self.session_id_prefix = session_id_prefix
        self.target_hand = target_hand
        self.allowed_splits = tuple(allowed_splits)
        self.evaluation_partition = load_evaluation_partition(
            evaluation_partition_path, evaluation_role, allowed_splits
        )
        if jitter and self.allowed_splits != ("train",):
            raise ValueError("Window jitter is supported only for the train split")
        self.jitter = bool(jitter)
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.emg_field = emg_field
        if norm_mode not in (None, "per-dataset"):
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")
        if norm_mode == "per-dataset" and norm_stats_path is None:
            raise ValueError("per-dataset normalization requires norm_stats_path")
        self.norm_mode = norm_mode
        self.transform = transform
        self.name = dataset_name
        self.max_open_sessions = max(1, int(max_open_sessions))
        self._cache: OrderedDict[int, dict[str, np.memmap]] = OrderedDict()
        self._sessions: list[_SessionHand] = []
        self._starts: np.ndarray
        self._session_indices: np.ndarray
        self._groups: list[np.ndarray] = []
        self._mean = np.zeros(8, dtype=np.float32)
        self._std = np.ones(8, dtype=np.float32)

        manifest = _read(self.dataset_dir / "dataset.json")
        if manifest.get("units", {}).get("emg") != "mV":
            raise ValueError("Session-v2 EMG must be declared in mV")
        if norm_mode == "per-dataset":
            self._load_stats(norm_stats_path)
        all_starts: list[np.ndarray] = []
        all_offsets: list[np.ndarray] = []
        all_sessions: list[np.ndarray] = []
        index_offset = 0
        for entry in manifest["sessions"]:
            session_path = self.dataset_dir / entry["path"]
            session = _read(session_path)
            if session_id_prefix is not None and not session["session_id"].startswith(
                session_id_prefix
            ):
                continue
            hand = session["hands"].get(target_hand)
            if not hand:
                continue
            arrays = hand["arrays"]
            for key in (*REQUIRED, emg_field, _valid_label_key(arrays)):
                if key not in arrays:
                    raise ValueError(f"Missing {key}: {session_path} {target_hand}")
            if arrays[emg_field].get("unit") != "mV":
                raise ValueError(f"EMG is not normalized to mV: {session_path}")
            record = _SessionHand(
                session_id=session["session_id"],
                source_group=entry["source_group"],
                subject=str(session["subject"]),
                folder=session_path.parent,
                arrays=arrays,
                rows=int(hand["rows"]),
            )
            starts, offsets = self._index_session(record)
            if not len(starts):
                continue
            session_index = len(self._sessions)
            self._sessions.append(record)
            all_starts.append(starts)
            all_offsets.append(offsets)
            all_sessions.append(np.full(len(starts), session_index, dtype=np.int16))
            self._groups.append(
                np.arange(index_offset, index_offset + len(starts), dtype=np.int64)
            )
            index_offset += len(starts)
        self._starts = (
            np.concatenate(all_starts) if all_starts else np.empty(0, dtype=np.int64)
        )
        self._session_indices = (
            np.concatenate(all_sessions)
            if all_sessions
            else np.empty(0, dtype=np.int16)
        )
        self._max_jitter_offsets = (
            np.concatenate(all_offsets) if all_offsets else np.empty(0, np.int64)
        )

    def _load_stats(self, path: str | Path | None) -> None:
        if path is None:
            return
        stats = _read(Path(path))
        key = f"{self.emg_field}_{self.target_hand}"
        item = stats[key]
        self._mean = np.asarray(item["mean"], dtype=np.float32)
        self._std = np.asarray(item["std"], dtype=np.float32)
        if (
            self._mean.shape != (8,)
            or self._std.shape != (8,)
            or np.any(self._std <= 0)
        ):
            raise ValueError(f"Invalid EMG normalization statistics for {key}")

    def _index_session(self, session: _SessionHand) -> tuple[np.ndarray, np.ndarray]:
        arrays = session.arrays
        folder = session.folder
        n = session.rows
        if n < self.window_length:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        split = np.asarray(_array(folder, arrays["split_id"]))
        timestamps = np.asarray(_array(folder, arrays["timestamps"]))
        allowed = np.isin(split, [SPLIT_IDS[name] for name in self.allowed_splits])
        if self.evaluation_partition is not None:
            gesture = (
                _array(folder, arrays["gesture_class"])
                if "gesture_class" in arrays
                else np.full(n, -1)
            )
            objects = (
                _array(folder, arrays["object_id"])
                if "object_id" in arrays
                else np.full(n, -1)
            )
            roles = self.evaluation_partition.classify(
                session.subject, split, gesture, objects, session_id=session.session_id
            )
            allowed &= roles == self.evaluation_partition.code
        if "emg_valid" in arrays:
            allowed &= np.asarray(_array(folder, arrays["emg_valid"]), dtype=bool)
        delta = np.diff(timestamps)
        boundary = (split[1:] != split[:-1]) | (delta < 0) | (delta > 100_000)
        for key in (
            "segment_id",
            "continuity_segment",
            "recording_index",
            "interaction_id",
        ):
            if key in arrays:
                values = np.asarray(_array(folder, arrays[key]))
                boundary |= values[1:] != values[:-1]
        boundary |= allowed[1:] != allowed[:-1]
        edges = np.r_[0, np.flatnonzero(boundary) + 1, n]
        starts, offsets = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if allowed[lo] and hi - lo >= self.window_length:
                grid = np.arange(lo, hi - self.window_length + 1, self.stride)
                starts.append(grid)
                offsets.append(
                    np.minimum(self.stride - 1, hi - grid - self.window_length)
                )
        if not starts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        candidates = np.concatenate(starts)
        label = np.asarray(
            _array(folder, arrays[_valid_label_key(arrays)]), dtype=np.int64
        )
        prefix = np.empty(n + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(label, out=prefix[1:])
        keep = prefix[candidates + self.window_length] > prefix[candidates]
        return candidates[keep], np.concatenate(offsets)[keep]

    def _session_arrays(self, index: int) -> dict[str, np.memmap]:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        record = self._sessions[index]
        needed = {
            self.emg_field,
            "joint_angles",
            "wrist_angles",
            "wrist_valid",
            "split_id",
            _valid_label_key(record.arrays),
        }
        needed.update(k for k in ("object_id",) if k in record.arrays)
        result = {key: _array(record.folder, record.arrays[key]) for key in needed}
        self._cache[index] = result
        if len(self._cache) > self.max_open_sessions:
            self._cache.popitem(last=False)
        return result

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self._starts)

    def session_index_groups(self) -> list[np.ndarray]:
        return self._groups

    def get_subject(self, index: int) -> str:
        return self._sessions[int(self._session_indices[index])].subject

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._read_into(index)

    @property
    def batch_read_signature(self) -> tuple:
        return ("session_emg", self.window_length)

    def _read_into(
        self, index: int, buffers: dict[str, np.ndarray] | None = None
    ) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        session_index = int(self._session_indices[index])
        record = self._sessions[session_index]
        arrays = self._session_arrays(session_index)
        start = int(self._starts[index])
        if self.jitter and self._max_jitter_offsets[index] > 0:
            shifted = start + int(
                np.random.randint(self._max_jitter_offsets[index] + 1)
            )
            # Never turn a supervised anchor into a completely unlabelled window.
            if np.any(
                arrays[_valid_label_key(record.arrays)][
                    shifted : shifted + self.window_length
                ]
            ):
                start = shifted
        end = start + self.window_length
        emg = np.array(arrays[self.emg_field][start:end], dtype=np.float32)
        emg -= self._mean
        emg /= self._std + np.float32(1e-6)
        if self.transform is not None:
            emg = self.transform({"emg": emg})
            if isinstance(emg, dict):
                emg = emg["emg"]
            if hasattr(emg, "numpy"):
                emg = emg.numpy()
        emg = np.asarray(emg, dtype=np.float32)
        if emg.shape != (self.window_length, 8):
            raise ValueError(f"EMG transform changed shape to {emg.shape}")
        # Fill the final channel-first buffer directly: avoid temporary full
        # sequences, concatenate, transpose-copy and three nan_to_num scans.
        target = (
            buffers["joint_angles"]
            if buffers is not None
            else np.empty((22, self.window_length), dtype=np.float32)
        )
        target[:20] = arrays["joint_angles"][start:end].T
        np.deg2rad(arrays["wrist_angles"][start:end].T, out=target[20:])
        label = np.asarray(
            arrays[_valid_label_key(record.arrays)][start:end], dtype=bool
        )
        wrist_valid = np.asarray(arrays["wrist_valid"][start:end], dtype=bool)
        mask = (
            buffers["label_valid_mask"]
            if buffers is not None
            else np.empty(target.shape, dtype=bool)
        )
        np.isfinite(target, out=mask)
        if not mask.all():
            target[~mask] = 0.0
        mask &= label[None, :]
        mask[20:] &= wrist_valid[None, :]
        split_id = (
            SPLIT_IDS[self.allowed_splits[0]] if len(self.allowed_splits) == 1 else None
        )
        if split_id is None:
            split_id = int(arrays["split_id"][start])
        if buffers is not None:
            buffers["emg"][:] = emg.T
            emg_result = buffers["emg"]
        else:
            emg_result = emg.T.copy()
        return {
            "emg": emg_result,
            "emg_start_row": start,
            "window_offset_samples": start - int(self._starts[index]),
            "emg_channel_mask": np.ones(8, dtype=bool),
            "joint_angles": target,
            "label_valid_mask": mask,
            "target_hand": self.target_hand,
            "target_hand_index": 0 if self.target_hand == "left" else 1,
            "dataset_name": "showee" if record.source_group == "showee" else "egoemg",
            "episode_id": record.session_id,
            "split_id": split_id,
            "object_id": (
                int(arrays["object_id"][start]) if "object_id" in arrays else -1
            ),
            "held_out_user": split_id in (1, 3, 5, 6, 7),
            "held_out_stage": split_id in (2, 3),
        }
