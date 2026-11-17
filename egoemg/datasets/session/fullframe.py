"""Full egocentric RGB -> simultaneous left/right angles, without crop geometry.

An offline builder publishes 256px RGB arrays and clock-aligned sidecars. Each
hand is masked independently, including split membership; one image can never
supervise a held-out hand through its training companion.
"""

from __future__ import annotations

from egoemg.datasets.session.alignment import dependency_signature

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from egoemg.datasets.evaluation_partition import load_evaluation_partition
from egoemg.datasets.session.schema import SPLIT_IDS, _array


class EgoEmgSessionFullFrameDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        allowed_splits,
        sample_interval_us=400_000,
        cache_dir=None,
        max_open_sessions=8,
        target_fps=None,
        session_ids=None,
        evaluation_partition_path=None,
        evaluation_role=None,
        **kwargs,
    ):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.cache_dir = Path(cache_dir or self.dataset_dir / "fullframe_256_index")
        self.allowed = np.array([SPLIT_IDS[x] for x in allowed_splits])
        self.evaluation_partition = load_evaluation_partition(
            evaluation_partition_path, evaluation_role, allowed_splits
        )
        self.max_open_sessions = max_open_sessions
        self.records = []
        self._cache = OrderedDict()
        self._groups = []
        indices = []
        offset = 0
        manifest = json.loads((self.dataset_dir / "dataset.json").read_text())
        for entry in manifest["sessions"]:
            path = self.dataset_dir / entry["path"]
            session = json.loads(path.read_text())
            if session_ids is not None and session["session_id"] not in session_ids:
                continue
            stem = self.cache_dir / session["session_id"]
            meta = stem.with_suffix(".json")
            if not meta.exists():
                raise RuntimeError(f"Prepare the cache first: {stem}")
            signature = dependency_signature(
                path, session, sample_interval_us, target_fps
            )
            if json.loads(meta.read_text())["signature"] != signature:
                raise RuntimeError(f"Stale cache: {stem}")
            with np.load(stem.with_suffix(".npz")) as data:
                alignment = {key: data[key].copy() for key in data.files}
            if self.evaluation_partition is not None:
                roles = np.full_like(alignment["splits"], -1)
                for h, side in enumerate(("left", "right")):
                    hand = session["hands"].get(side)
                    if not hand:
                        continue
                    arrays = hand["arrays"]
                    rows = alignment["rows"][:, h]
                    gesture = (
                        _array(path.parent, arrays["gesture_class"])[rows]
                        if "gesture_class" in arrays
                        else np.full(len(rows), -1)
                    )
                    objects = (
                        _array(path.parent, arrays["object_id"])[rows]
                        if "object_id" in arrays
                        else np.full(len(rows), -1)
                    )
                    roles[:, h] = self.evaluation_partition.classify(
                        session["subject"],
                        alignment["splits"][:, h],
                        gesture,
                        objects,
                        session_id=session["session_id"],
                    )
                crossed = (roles == 1).any(1) & (roles == 2).any(1)
                roles[crossed] = -1
                alignment["evaluation_roles"] = roles
            keep = np.flatnonzero(self._selected_hands(alignment).any(1))
            if not len(keep):
                continue
            record = len(self.records)
            self.records.append(
                dict(
                    folder=path.parent, session=session, stem=stem, alignment=alignment
                )
            )
            indices.append(np.column_stack((np.full(len(keep), record), keep)))
            self._groups.append(np.arange(offset, offset + len(keep)))
            offset += len(keep)
        self.samples = (
            np.concatenate(indices) if indices else np.empty((0, 2), np.int64)
        )
        self.batch_read_signature = ("fullframe_bimanual_256",)

    def __len__(self):
        return len(self.samples)

    def _selected_hands(self, alignment):
        selected = np.isin(alignment["splits"], self.allowed)
        if self.evaluation_partition is not None:
            selected &= alignment["evaluation_roles"] == self.evaluation_partition.code
        return selected

    def __getitem__(self, index):
        return self._read_into(index)

    def _read_into(self, index, buffers=None):
        result = self._read_targets(index)
        record_id, local = self.samples[index]
        cached = self._cache[record_id]
        if "rgb" not in cached:
            cached["rgb"] = np.load(
                self.records[record_id]["stem"].with_suffix(".npy"), mmap_mode="r"
            )
        result["vision_img"] = cached["rgb"][local].transpose(2, 0, 1).copy()
        if buffers is not None:
            for key, buffer in buffers.items():
                buffer[:] = result[key]
                result[key] = buffer
        return result

    def _read_targets(self, index):
        """Read bimanual targets without opening an image cache."""
        record_id, local = self.samples[index]
        record = self.records[record_id]
        if record_id not in self._cache:
            hand_arrays = {}
            for name, hand in record["session"]["hands"].items():
                if hand:
                    hand_arrays[name] = {
                        key: _array(record["folder"], spec)
                        for key, spec in hand["arrays"].items()
                        if key
                        in {"joint_angles", "wrist_angles", "wrist_valid", "object_id"}
                    }
            self._cache[record_id] = dict(
                hands=hand_arrays,
            )
            while len(self._cache) > self.max_open_sessions:
                self._cache.popitem(last=False)
        self._cache.move_to_end(record_id)
        cached = self._cache[record_id]
        alignment = record["alignment"]
        angles = np.zeros((2, 22, 1), np.float32)
        masks = np.zeros((2, 22, 1), bool)
        objects = np.full(2, -1, np.int64)
        session = record["session"]
        for hand_index, hand_name in enumerate(("left", "right")):
            hand = session["hands"].get(hand_name)
            if not hand:
                continue
            arrays = cached["hands"][hand_name]
            row = alignment["rows"][local, hand_index]
            target = np.concatenate(
                (
                    arrays["joint_angles"][row],
                    np.deg2rad(arrays["wrist_angles"][row]),
                )
            )
            valid = np.isfinite(target)
            valid &= alignment["splits"][local, hand_index] in self.allowed
            if self.evaluation_partition is not None:
                valid &= (
                    alignment["evaluation_roles"][local, hand_index]
                    == self.evaluation_partition.code
                )
            valid[20:] &= bool(arrays["wrist_valid"][row])
            if session["source_group"] == "showee":
                valid[20:] = False
            angles[hand_index, :, 0] = np.nan_to_num(target)
            masks[hand_index, :, 0] = valid
            if "object_id" in arrays:
                objects[hand_index] = arrays["object_id"][row]
        result = dict(
            emg=np.zeros((8, 1), np.float32),
            joint_angles=angles.reshape(44, 1),
            label_valid_mask=masks.reshape(44, 1),
        )
        result.update(
            dataset_name="showee" if session["source_group"] == "showee" else "egoemg",
            episode_id=session["session_id"],
            camera_frame=int(alignment["frames"][local]),
            split_id=alignment["splits"][local].astype(np.int64),
            object_id=objects,
        )
        return result
