"""Shared physical boundaries and evaluation roles for aligned EMG windows."""

from __future__ import annotations

from typing import Any

import numpy as np

from egoemg.datasets.session.schema import _array


def evaluation_roles(folder, arrays, subject, session_id, partition):
    split = _array(folder, arrays["split_id"])
    gesture = (
        _array(folder, arrays["gesture_class"])
        if "gesture_class" in arrays
        else np.full(len(split), -1)
    )
    objects = (
        _array(folder, arrays["object_id"])
        if "object_id" in arrays
        else np.full(len(split), -1)
    )
    return partition.classify(subject, split, gesture, objects, session_id)


def valid_emg_windows(
    record: dict[str, Any],
    rows: np.ndarray,
    *,
    window_length: int,
    emg_field: str,
    evaluation_partition=None,
    cache_boundaries: bool = False,
) -> np.ndarray:
    """Reject windows crossing splits, clocks, recordings or invalid EMG spans."""
    folder, arrays = record["folder"], record["arrays"]
    count = int(arrays[emg_field]["shape"][0])
    starts = rows - window_length // 2
    ends = starts + window_length
    valid = (starts >= 0) & (ends <= count)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return valid
    starts, ends = starts[indices], ends[indices]
    # Anchor filtering and dense jitter candidates share the same physical
    # boundaries. Scan each recording once during construction.
    edges = record.get("jitter_edges")
    if edges is None:
        times = np.asarray(_array(folder, arrays["timestamps"]))
        delta = np.diff(times)
        boundaries = (delta < 0) | (delta > 100_000)
        if evaluation_partition is not None:
            roles = evaluation_roles(
                folder,
                arrays,
                record["subject"],
                record["session_id"],
                evaluation_partition,
            )
            boundaries |= roles[1:] != roles[:-1]
        for key in (
            "split_id",
            "segment_id",
            "continuity_segment",
            "recording_index",
            "interaction_id",
            "emg_valid",
        ):
            if key in arrays:
                values = _array(folder, arrays[key])
                boundaries |= values[1:] != values[:-1]
        edges = np.flatnonzero(boundaries) + 1
        if cache_boundaries:
            record["jitter_edges"] = edges
    same_segment = np.searchsorted(edges, starts, side="right") == np.searchsorted(
        edges, ends - 1, side="right"
    )
    if "emg_valid" in arrays:
        prefix = record.get("jitter_invalid_prefix")
        if prefix is None:
            invalid = ~np.asarray(_array(folder, arrays["emg_valid"]), dtype=bool)
            prefix = np.r_[0, np.cumsum(invalid, dtype=np.int64)]
            if cache_boundaries:
                record["jitter_invalid_prefix"] = prefix
        same_segment &= prefix[ends] == prefix[starts]
    valid[indices] = same_segment
    return valid


def read_emg_window(values, start, length, mean, std, transform=None):
    """Normalize a writable (time, 8) window and preserve its time/channel axes."""
    signal = np.array(values[start : start + length], dtype=np.float32)
    signal -= mean
    signal /= std + np.float32(1e-6)
    if transform is not None:
        signal = transform({"emg": signal})
        if isinstance(signal, dict):
            signal = signal["emg"]
        if hasattr(signal, "numpy"):
            signal = signal.numpy()
    signal = np.asarray(signal, np.float32)
    if signal.shape != (length, 8):
        raise ValueError(
            f"EMG transform must preserve ({length}, 8), got {signal.shape}"
        )
    return signal
