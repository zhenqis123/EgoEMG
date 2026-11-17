from __future__ import annotations

from egoemg.datasets.session.windows import read_emg_window, valid_emg_windows
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
from egoemg.datasets.session.schema import _array
from egoemg.datasets.session.crops import (
    EgoEmgSessionVisionDataset,
    EgoEmgSessionCropPairDataset,
)
from egoemg.datasets.session.fullframe import EgoEmgSessionFullFrameDataset
from egoemg.datasets.session.temporal import (
    EgoEmgSessionTemporalVisionDataset,
)


class EgoEmgSessionFusionDataset(EgoEmgSessionVisionDataset):
    """Keep the vision target and attach EMG centered on its aligned hand row.

    A window must remain inside one split, recording, clock segment and valid
    EMG span. Labels and RGB retain the existing vision-reader semantics.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        allowed_splits: Sequence[str],
        window_length: int = 8_000,
        emg_field: str = "emg_filtered",
        norm_mode: str | None = "per-dataset",
        norm_stats_path: str | Path | None = None,
        transform: Any = None,
        jitter: bool = False,
        **kwargs: Any,
    ) -> None:
        if window_length <= 0 or emg_field not in {"emg_filtered", "emg_raw"}:
            raise ValueError("Require a positive window and a known EMG field")
        if norm_mode not in (None, "per-dataset"):
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")
        if norm_mode == "per-dataset" and norm_stats_path is None:
            raise ValueError("per-dataset normalization requires norm_stats_path")
        self.window_length = int(window_length)
        if jitter and tuple(allowed_splits) != ("train",):
            raise ValueError("Window jitter is supported only for the train split")
        self.jitter = bool(jitter)
        self.sample_interval_us = int(kwargs.get("sample_interval_us", 400_000))
        if self.jitter and self.sample_interval_us <= 0:
            raise ValueError("Fusion jitter requires a positive sample_interval_us")
        self.emg_field = emg_field
        self.norm_mode = norm_mode
        self._emg_stats = (
            json.loads(Path(norm_stats_path).read_text())
            if norm_mode == "per-dataset"
            else None
        )
        super().__init__(dataset_dir, allowed_splits, transform=transform, **kwargs)
        manifest = json.loads((self.dataset_dir / "dataset.json").read_text())
        if manifest.get("units", {}).get("emg") != "mV":
            raise ValueError("Session-v2 EMG must be declared in mV")
        selected = []
        jitter_ranges = []
        self._groups = []
        offset = 0
        for rec_id, record in enumerate(self.records):
            arrays = record["arrays"]
            spec = arrays.get(emg_field)
            if spec is None or spec.get("unit") != "mV":
                raise ValueError(f"Missing mV {emg_field}: {record['session_id']}")
            if tuple(spec["shape"])[1:] != (8,):
                raise ValueError("Session fusion requires eight EMG channels")
            mean, std = np.zeros(8, np.float32), np.ones(8, np.float32)
            if self._emg_stats is not None:
                stats = self._emg_stats[f"{emg_field}_{record['hand']}"]
                mean = np.asarray(stats["mean"], np.float32)
                std = np.asarray(stats["std"], np.float32)
                if mean.shape != (8,) or std.shape != (8,) or np.any(std <= 0):
                    raise ValueError("Invalid EMG normalization statistics")
            record["emg_mean"], record["emg_std"] = mean, std
            indices = np.flatnonzero(self.samples[:, 0] == rec_id)
            rows = self.samples[indices, 2]
            valid = self._valid_windows(record, rows)
            indices = indices[valid]
            for item in self.coverage:
                if (
                    item["session_id"] == record["session_id"]
                    and item["hand"] == record["hand"]
                ):
                    item["fusion_samples"] = len(indices)
            if len(indices):
                selected.append(self.samples[indices])
                if self.jitter:
                    jitter_ranges.append(
                        self._index_jitter(record, self.samples[indices])
                    )
                self._groups.append(np.arange(offset, offset + len(indices)))
                offset += len(indices)
            else:
                record.pop("jitter_edges", None)
                record.pop("jitter_invalid_prefix", None)
        self.samples = (
            np.concatenate(selected) if selected else np.empty((0, 4), np.int64)
        )
        self.name = "egoemg_session_v2_fusion"
        self._jitter_ranges = (
            np.concatenate(jitter_ranges)
            if jitter_ranges
            else np.empty((0, 2), np.int64)
        )

    def _valid_windows(self, record, rows):
        return valid_emg_windows(
            record,
            rows,
            window_length=self.window_length,
            emg_field=self.emg_field,
            evaluation_partition=self.evaluation_partition,
            cache_boundaries=self.jitter,
        )

    def _index_jitter(self, record: dict[str, Any], anchors: np.ndarray) -> np.ndarray:
        """Group legal dense frames by time bin and physical block.

        Retain the original sparse anchors and their weights. Only their aligned
        frame/label/EMG realization changes on each training access.
        """
        path = self.index_dir / f"{record['session_id']}_{record['hand']}.npz"
        with np.load(path) as index:
            frames, rows = index["frames"], index["rows"]
            splits, times = index["splits"], index["times"]
        valid = (splits == 0) & self._valid_windows(record, rows)
        candidates = np.column_stack((frames[valid], rows[valid], splits[valid]))
        edges = record.pop("jitter_edges")
        record.pop("jitter_invalid_prefix", None)
        keys = np.column_stack(
            (
                times[valid] // self.sample_interval_us,
                np.searchsorted(edges, rows[valid], side="right"),
            )
        )
        positions = np.searchsorted(frames, anchors[:, 1])
        anchor_keys = np.column_stack(
            (
                times[positions] // self.sample_interval_us,
                np.searchsorted(edges, anchors[:, 2], side="right"),
            )
        )
        _, groups = np.unique(
            np.concatenate((keys, anchor_keys)), axis=0, return_inverse=True
        )
        candidate_groups, anchor_groups = groups[: len(keys)], groups[len(keys) :]
        order = np.argsort(candidate_groups, kind="stable")
        record["jitter_candidates"] = candidates[order]
        sorted_groups = candidate_groups[order]
        ranges = np.column_stack(
            (
                np.searchsorted(sorted_groups, anchor_groups, side="left"),
                np.searchsorted(sorted_groups, anchor_groups, side="right"),
            )
        )
        if np.any(ranges[:, 0] >= ranges[:, 1]):
            raise RuntimeError("A fusion jitter anchor has no legal aligned frame")
        return ranges

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._read_into(index)

    @property
    def batch_read_signature(self) -> tuple:
        return ("session_fusion", self.window_length, self.return_uint8)

    def _read_into(
        self, index: int, buffers: dict[str, np.ndarray] | None = None
    ) -> dict[str, Any]:
        rec_id, frame, row, split = self.samples[index]
        anchor_row, anchor_frame = int(row), int(frame)
        record = self.records[rec_id]
        if self.jitter:
            lo, hi = self._jitter_ranges[index]
            frame, row, split = record["jitter_candidates"][np.random.randint(lo, hi)]
        sample = self._get_sample(rec_id, frame, row, split, buffers=buffers)
        values = self._cache[rec_id]
        if self.emg_field not in values:
            values[self.emg_field] = _array(
                record["folder"], record["arrays"][self.emg_field]
            )
        start = int(row) - self.window_length // 2
        emg = read_emg_window(
            values[self.emg_field],
            start,
            self.window_length,
            record["emg_mean"],
            record["emg_std"],
            self.transform,
        )
        if buffers is not None:
            buffers["emg"][:] = emg.T
            emg_result = buffers["emg"]
        else:
            emg_result = emg.T.copy()
        sample.update(
            emg=emg_result,
            emg_channel_mask=np.ones(8, dtype=bool),
            emg_start_row=start,
            vision_valid_mask=np.bool_(True),
            window_offset_samples=int(row) - anchor_row,
            anchor_camera_frame=anchor_frame,
            anchor_label_row=anchor_row,
        )
        return sample


class EgoEmgSessionFullFrameFusionDataset(EgoEmgSessionFullFrameDataset):
    """Preserve the vision pool; unavailable EMG gets a per-hand false mask.

    Jitter moves the window start, not the image or its target. The reference
    camera/pose remains fixed and inside the 4s window. Offsets are uniform
    within +/- half the 400ms sampling stride, clipped to legal physical bounds.
    """

    def __init__(
        self,
        dataset_dir,
        allowed_splits,
        window_length=8000,
        emg_field="emg_filtered",
        norm_stats_path=None,
        norm_mode="per-dataset",
        transform=None,
        jitter=False,
        emg_past_only=False,
        **kwargs,
    ):
        if window_length <= 0 or emg_field != "emg_filtered":
            raise ValueError("Require positive window length and filtered EMG")
        if jitter and list(allowed_splits) != ["train"]:
            raise ValueError("Window jitter is training-only")
        if norm_mode != "per-dataset" or norm_stats_path is None:
            raise ValueError("Fusion requires training-only per-hand statistics")
        self.window_length = int(window_length)
        self.emg_field = emg_field
        self.jitter = jitter
        self.emg_past_only = bool(emg_past_only)
        if self.emg_past_only and jitter:
            raise ValueError("Past-only fusion requires fixed endpoint EMG windows")
        self.transform = transform
        super().__init__(dataset_dir, allowed_splits, **kwargs)
        self.sample_interval_us = int(kwargs.get("sample_interval_us", 400000))
        if jitter and self.sample_interval_us <= 0:
            raise ValueError("Window jitter requires a positive sampling interval")
        self.jitter_radius = self.sample_interval_us * 2000 // 2_000_000
        stats = json.loads(Path(norm_stats_path).read_text())
        for record in self.records:
            alignment = record["alignment"]
            record["emg_ranges"] = np.full(
                (len(alignment["frames"]), 2, 2), -1, np.int64
            )
            record["emg_stats"] = {}
            for index, side in enumerate(("left", "right")):
                hand = record["session"]["hands"].get(side)
                if not hand:
                    continue
                arrays = hand["arrays"]
                spec = arrays[self.emg_field]
                if spec.get("unit") != "mV" or spec["shape"][1:] != [8]:
                    raise ValueError("Expected 8-channel mV EMG")
                stat = stats[f"{self.emg_field}_{side}"]
                mean, std = (
                    np.asarray(stat[key], np.float32) for key in ("mean", "std")
                )
                if mean.shape != (8,) or std.shape != (8,) or np.any(std <= 0):
                    raise ValueError("Invalid per-hand EMG statistics")
                record["emg_stats"][side] = (mean, std)
                rows = alignment["rows"][:, index]
                boundary_record = dict(
                    folder=record["folder"],
                    arrays=arrays,
                    subject=record["session"].get("subject"),
                    session_id=record["session"]["session_id"],
                )
                checker_rows = (
                    rows - self.window_length + 1 + self.window_length // 2
                    if self.emg_past_only
                    else rows
                )
                valid = valid_emg_windows(
                    boundary_record,
                    checker_rows,
                    window_length=self.window_length,
                    emg_field=self.emg_field,
                    evaluation_partition=self.evaluation_partition,
                    cache_boundaries=True,
                )
                valid &= self._selected_hands(alignment)[:, index]
                if "jitter_edges" not in boundary_record:
                    continue
                edges = boundary_record["jitter_edges"]
                limits = np.r_[0, edges, spec["shape"][0]]
                block = np.searchsorted(edges, rows, side="right")
                starts = (
                    rows - self.window_length + 1
                    if self.emg_past_only
                    else rows - self.window_length // 2
                )
                radius = self.jitter_radius if jitter else 0
                low = np.maximum(starts - radius, limits[block])
                high = np.minimum(
                    starts + radius, limits[block + 1] - self.window_length
                )
                low = np.maximum(low, rows - self.window_length + 1)
                high = np.minimum(high, rows)
                valid &= low <= high
                record["emg_ranges"][valid, index] = np.column_stack(
                    (low[valid], high[valid])
                )
        self.batch_read_signature = ("fullframe_bimanual_fusion", self.window_length)

    def _read_into(self, index, buffers=None):
        # The vision reader transports a dummy (8,1) EMG field; replace it only
        # after reading, so shared buffers can have the real (2,8,WL) layout.
        sample = super()._read_into(index)
        record_id, local = self.samples[index]
        record = self.records[record_id]
        cached = self._cache[record_id]
        emg = np.zeros((2, 8, self.window_length), np.float32)
        available = np.zeros(2, bool)
        starts = np.full(2, -1, np.int64)
        offsets = np.zeros(2, np.int64)
        last_times = np.full(2, -1, np.int64)
        for hand_index, side in enumerate(("left", "right")):
            low, high = record["emg_ranges"][local, hand_index]
            if low < 0:
                continue
            start = int(np.random.randint(low, high + 1)) if self.jitter else int(low)
            values = cached["hands"][side]
            if self.emg_field not in values:
                spec = record["session"]["hands"][side]["arrays"][self.emg_field]
                values[self.emg_field] = _array(record["folder"], spec)
            mean, std = record["emg_stats"][side]
            signal = read_emg_window(
                values[self.emg_field],
                start,
                self.window_length,
                mean,
                std,
                self.transform,
            )
            emg[hand_index] = signal.T
            available[hand_index] = True
            starts[hand_index] = start
            if "timestamps" not in values:
                values["timestamps"] = _array(
                    record["folder"],
                    record["session"]["hands"][side]["arrays"]["timestamps"],
                )
            last_times[hand_index] = values["timestamps"][
                start + self.window_length - 1
            ]
            if (
                self.emg_past_only
                and last_times[hand_index] > record["alignment"]["times"][local]
            ):
                raise ValueError("Future EMG entered a past-only window")
            offsets[hand_index] = start - (
                record["alignment"]["rows"][local, hand_index]
                - (
                    self.window_length - 1
                    if self.emg_past_only
                    else self.window_length // 2
                )
            )
        sample["emg"] = emg
        if buffers is not None:
            for key, value in buffers.items():
                value[:] = sample[key]
                sample[key] = value
        sample.update(
            emg_valid_mask=available,
            emg_start_row=starts,
            emg_last_timestamp_us=last_times,
            window_offset_samples=offsets,
            anchor_timestamp_us=np.int64(record["alignment"]["times"][local]),
        )
        return sample


class EgoEmgSessionTemporalFusionDataset(
    EgoEmgSessionTemporalVisionDataset, EgoEmgSessionFullFrameFusionDataset
):
    """Cooperative readers preserve center samples, EMG masks and shared batches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_read_signature = (
            "temporal_fullframe_bimanual_fusion",
            self.num_frames,
            self.window_length,
        )
        if self.causal_context or self.emg_past_only:
            self.batch_read_signature += (
                "causal",
                self.causal_context,
                self.emg_past_only,
            )


class EgoEmgSessionCropPairFusionDataset(
    EgoEmgSessionFullFrameFusionDataset, EgoEmgSessionCropPairDataset
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_read_signature = ("native_crop_pair_fusion", self.window_length)

    def _read_into(self, index, buffers=None):
        sample = super()._read_into(index, buffers)
        supervised = sample["label_valid_mask"].reshape(2, 22, 1).any((1, 2))
        sample["emg_valid_mask"] &= sample["crop_valid"] & supervised
        return sample
