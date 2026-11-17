from __future__ import annotations

from egoemg.datasets.session.alignment import align_camera_rows, ego_stream

from egoemg.datasets.session.windows import evaluation_roles
import json
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from egoemg.datasets.session.schema import SPLIT_IDS, _array, _valid_label_key
from egoemg.datasets.session.fullframe import EgoEmgSessionFullFrameDataset


class EgoEmgSessionVisionDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        allowed_splits,
        sample_interval_us=400_000,
        index_dir=None,
        auto_build_index=False,
        return_uint8=False,
        max_open_sessions=128,
        transform=None,
        norm_mode=None,
        norm_stats_path=None,
        evaluation_partition_path=None,
        evaluation_role=None,
    ):
        from egoemg.datasets.evaluation_partition import load_evaluation_partition

        self.evaluation_partition = load_evaluation_partition(
            evaluation_partition_path, evaluation_role, allowed_splits
        )
        self.dataset_dir = Path(dataset_dir).resolve()
        if not allowed_splits or sample_interval_us < 0 or max_open_sessions < 1:
            raise ValueError(
                "Require splits, a nonnegative interval and a positive cache"
            )
        self.index_dir = Path(index_dir or self.dataset_dir / "vision_crop_index")
        self.return_uint8 = return_uint8
        self.max_open_sessions = max_open_sessions
        self.transform = transform
        self.name = "egoemg_session_v2_vision"
        self._cache = OrderedDict()
        self.records = []
        self._groups = []
        self.coverage = []
        indices = []
        offset = 0
        allowed = [SPLIT_IDS[name] for name in allowed_splits]
        manifest = json.loads((self.dataset_dir / "dataset.json").read_text())
        for entry in manifest["sessions"]:
            session_path = self.dataset_dir / entry["path"]
            session = json.loads(session_path.read_text())
            folder = session_path.parent
            cameras = session["cameras"]
            device = "showee_head" if cameras.get("showee_head") else "gopro"
            stream_name = "camera" if device == "showee_head" else "rgb"
            stream = (cameras.get(device) or {}).get("streams", {}).get(stream_name)
            if not stream:
                raise ValueError(f"Missing egocentric stream: {session_path}")
            crops = stream.get("hand_crops", {})
            variant = next(
                (
                    v
                    for v in crops.get("variants", [])
                    if v["crop_id"] == crops.get("active_id")
                ),
                None,
            )
            if not variant or not variant.get("generation_complete"):
                raise ValueError(f"Incomplete hand crops: {session_path}")
            for hand_index, hand in enumerate(("left", "right")):
                payload = session["hands"].get(hand)
                if not payload:
                    continue
                arrays = payload["arrays"]
                files = [session_path, folder / variant["storage"]["path"]]
                for specs, keys in (
                    (stream["arrays"], ("timestamps", "timestamp_valid")),
                    (variant["arrays"], ("valid", "offsets")),
                    (
                        arrays,
                        (
                            "timestamps",
                            "split_id",
                            "joint_angles",
                            _valid_label_key(arrays),
                        ),
                    ),
                ):
                    files.extend(folder / specs[k]["path"] for k in keys if k in specs)
                signature = {
                    str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files
                }
                signature["index_version"] = 2
                stem = f"{session['session_id']}_{hand}"
                metadata_path = self.index_dir / (stem + ".json")
                index_path = self.index_dir / (stem + ".npz")
                fresh = (
                    metadata_path.exists()
                    and index_path.exists()
                    and json.loads(metadata_path.read_text()) == signature
                )
                if not fresh:
                    if not auto_build_index:
                        raise RuntimeError(
                            f"Missing/stale crop index: {stem}; "
                            "run benchmark with --build-index"
                        )
                    times = _array(folder, stream["arrays"]["timestamps"])
                    hand_times = _array(folder, arrays["timestamps"])
                    rows, time_valid, ages = align_camera_rows(hand_times, times)
                    valid = np.asarray(
                        _array(folder, variant["arrays"]["valid"])[:, hand_index],
                        dtype=bool,
                    ).copy()
                    valid &= time_valid
                    if "timestamp_valid" in stream["arrays"]:
                        valid &= _array(
                            folder, stream["arrays"]["timestamp_valid"]
                        ).astype(bool)
                    valid &= _array(folder, arrays[_valid_label_key(arrays)])[
                        rows
                    ].astype(bool)
                    valid &= np.isfinite(
                        _array(folder, arrays["joint_angles"])[rows]
                    ).all(1)
                    split = _array(folder, arrays["split_id"])[rows]
                    valid &= np.isin(split, list(SPLIT_IDS.values()))
                    offsets = _array(folder, variant["arrays"]["offsets"])[
                        :, hand_index
                    ]
                    valid &= offsets[:, 1] > 0
                    valid &= (
                        offsets.sum(1)
                        <= (folder / variant["storage"]["path"]).stat().st_size
                    )
                    frames = np.flatnonzero(valid)
                    self.index_dir.mkdir(parents=True, exist_ok=True)
                    temp = index_path.with_suffix(f".{os.getpid()}.tmp")
                    with temp.open("wb") as f:
                        np.savez(
                            f,
                            frames=frames,
                            rows=rows[frames],
                            splits=split[frames],
                            times=times[frames],
                            ages=ages[frames],
                        )
                    os.replace(temp, index_path)
                    temp_meta = metadata_path.with_suffix(f".{os.getpid()}.tmp")
                    temp_meta.write_text(json.dumps(signature, sort_keys=True))
                    os.replace(temp_meta, metadata_path)
                with np.load(index_path) as index:
                    keep = np.flatnonzero(np.isin(index["splits"], allowed))
                    if self.evaluation_partition is not None:
                        rows = index["rows"][keep]
                        roles = self._evaluation_roles(
                            folder, arrays, session["subject"], session["session_id"]
                        )
                        keep = keep[roles[rows] == self.evaluation_partition.code]
                    if len(keep) and sample_interval_us:
                        # Uniform temporal bins; do not let camera FPS alter weighting.
                        bins = index["times"][keep] // sample_interval_us
                        keep = keep[np.r_[True, bins[1:] != bins[:-1]]]
                    frames, rows = index["frames"][keep], index["rows"][keep]
                    splits = index["splits"][keep]
                    self.coverage.append(
                        dict(
                            session_id=session["session_id"],
                            hand=hand,
                            samples=len(keep),
                            max_age_us=(
                                int(index["ages"][keep].max()) if len(keep) else None
                            ),
                        )
                    )
                if not len(frames):
                    continue
                record_index = len(self.records)
                self.records.append(
                    dict(
                        folder=folder,
                        arrays=arrays,
                        variant=variant,
                        hand_index=hand_index,
                        hand=hand,
                        source_group=entry["source_group"],
                        session_id=session["session_id"],
                        subject=session.get("subject"),
                    )
                )
                indices.append(
                    np.column_stack(
                        (np.full(len(frames), record_index), frames, rows, splits)
                    )
                )
                self._groups.append(np.arange(offset, offset + len(frames)))
                offset += len(frames)
        self.samples = (
            np.concatenate(indices).astype(np.int64)
            if indices
            else np.empty((0, 4), np.int64)
        )
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

    def _evaluation_roles(self, folder, arrays, subject, session_id):
        return evaluation_roles(
            folder, arrays, subject, session_id, self.evaluation_partition
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _worker_init(_):
        import cv2
        import torch

        cv2.setNumThreads(1)
        torch.set_num_threads(1)

    def session_index_groups(self):
        return self._groups

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def __getitem__(self, index):
        return self._get_sample(*self.samples[index])

    @property
    def batch_read_signature(self):
        return ("session_vision", self.return_uint8)

    def _read_into(self, index, buffers=None):
        return self._get_sample(*self.samples[index], buffers=buffers)

    def _get_sample(self, rec_id, frame, row, split, buffers=None):
        import cv2

        record = self.records[rec_id]
        if rec_id not in self._cache:
            folder, arrays, variant = (
                record["folder"],
                record["arrays"],
                record["variant"],
            )
            self._cache[rec_id] = {
                "data": np.memmap(
                    str(folder / variant["storage"]["path"]), np.uint8, mode="r"
                ),
                "offsets": _array(folder, variant["arrays"]["offsets"]),
                **{
                    k: _array(folder, arrays[k])
                    for k in (
                        "joint_angles",
                        "wrist_angles",
                        "wrist_valid",
                        "object_id",
                    )
                    if k in arrays
                },
            }
            while len(self._cache) > self.max_open_sessions:
                self._cache.popitem(last=False)
        self._cache.move_to_end(rec_id)
        values = self._cache[rec_id]
        offset, length = map(int, values["offsets"][frame, record["hand_index"]])
        rgb_flag = getattr(cv2, "IMREAD_COLOR_RGB", None)
        image = cv2.imdecode(
            values["data"][offset : offset + length],
            rgb_flag if rgb_flag is not None else cv2.IMREAD_COLOR,
        )
        if image is None or image.shape != (256, 256, 3):
            raise ValueError(f"Bad crop: {record['session_id']} frame={frame}")
        if rgb_flag is None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Preserve contiguous single-sample outputs. Batch reads instead write
        # directly into final shared CHW storage, without a private CHW copy.
        image = image.transpose(2, 0, 1)
        if buffers is not None:
            buffers["vision_img"][:] = image
            image = buffers["vision_img"]
        else:
            image = image.copy()
        if not self.return_uint8:
            image = image.astype(np.float32, copy=False)
            image /= 255
            image -= self.mean
            image /= self.std
        angles = np.concatenate(
            (values["joint_angles"][row], np.deg2rad(values["wrist_angles"][row]))
        ).astype(np.float32)
        mask = np.isfinite(angles)
        mask[20:] &= (
            bool(values["wrist_valid"][row]) and record["source_group"] != "showee"
        )
        angles = np.nan_to_num(angles)[:, None]
        mask = mask[:, None]
        emg = np.zeros((8, 1), np.float32)
        if buffers is not None:
            for key, value in (
                ("vision_img", image),
                ("joint_angles", angles),
                ("label_valid_mask", mask),
            ):
                if buffers[key] is not value:
                    buffers[key][:] = value
            image = buffers["vision_img"]
            angles = buffers["joint_angles"]
            mask = buffers["label_valid_mask"]
            # Fusion fills its full EMG window after this shared image path.
            if buffers["emg"].shape == (8, 1):
                buffers["emg"].fill(0)
                emg = buffers["emg"]
        return dict(
            vision_img=image,
            joint_angles=angles,
            label_valid_mask=mask,
            emg=emg,
            dataset_name="showee" if record["source_group"] == "showee" else "egoemg",
            target_hand_index=record["hand_index"],
            episode_id=record["session_id"],
            camera_frame=int(frame),
            label_row=int(row),
            split_id=int(split),
            object_id=int(values["object_id"][row]) if "object_id" in values else -1,
        )


class EgoEmgSessionCropPairDataset(EgoEmgSessionFullFrameDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crop_cache = {}
        good = np.zeros(len(self.samples), bool)
        for rid, record in enumerate(self.records):
            stream = ego_stream(record["session"])
            crops = stream["hand_crops"]
            variant = next(
                v for v in crops["variants"] if v["crop_id"] == crops["active_id"]
            )
            arrays = variant["arrays"]
            valid = _array(record["folder"], arrays["valid"])
            positions = np.flatnonzero(self.samples[:, 0] == rid)
            local = self.samples[positions, 1]
            frames = record["alignment"]["frames"][local]
            selected = self._selected_hands(record["alignment"])[local]
            good[positions] = (valid[frames].astype(bool) & selected).any(1)
            record["crop_variant"] = variant
        self.samples = self.samples[good]
        self._groups = [
            np.flatnonzero(self.samples[:, 0] == rid)
            for rid in range(len(self.records))
        ]
        self._groups = [g for g in self._groups if len(g)]
        self.batch_read_signature = ("native_crop_pair_256",)

    def _read_into(self, index, buffers=None):
        import cv2

        result = self._read_targets(index)
        rid, local = self.samples[index]
        record = self.records[rid]
        frame = int(record["alignment"]["frames"][local])
        if rid not in self._crop_cache:
            variant = record["crop_variant"]
            self._crop_cache[rid] = dict(
                valid=_array(record["folder"], variant["arrays"]["valid"]),
                offsets=_array(record["folder"], variant["arrays"]["offsets"]),
                path=record["folder"] / variant["storage"]["path"],
            )
        cached = self._crop_cache[rid]
        images = np.zeros((2, 3, 256, 256), np.uint8)
        crop_valid = cached["valid"][frame].astype(bool)
        with cached["path"].open("rb") as handle:
            for side in range(2):
                if not crop_valid[side]:
                    continue
                offset, length = map(int, cached["offsets"][frame, side])
                handle.seek(offset)
                image = cv2.imdecode(
                    np.frombuffer(handle.read(length), np.uint8), cv2.IMREAD_COLOR
                )
                if image is None or image.shape != (256, 256, 3):
                    raise ValueError(
                        f'Invalid native crop: {record["session"]["session_id"]} {frame} {side}'
                    )
                images[side] = image[:, :, ::-1].transpose(2, 0, 1)
        result["vision_img"] = images
        result["label_valid_mask"] = (
            result["label_valid_mask"].reshape(2, 22, 1) & crop_valid[:, None, None]
        ).reshape(44, 1)
        if buffers is not None:
            for key, value in buffers.items():
                value[:] = result[key]
                result[key] = value
        result["crop_valid"] = crop_valid
        result["physical_frame"] = int(record["alignment"]["physical"][local])
        return result
