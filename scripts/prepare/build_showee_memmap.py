"""Convert raw ShowEE session/task folders to EgoEMG-compatible memmaps.

The converter is intentionally shard-oriented: write a new memmap directory
instead of mutating an existing EgoEMG memmap.  A typical smoke test is::

    python scripts/prepare/build_showee_memmap.py \
        --source-root data/showee \
        --out-root data/ShowEE_202607_memmap_smoke \
        --episode 20260714_0062_midair_1/thumb_middle \
        --episode 20260714_0062_handobject_1/pick_up_put_down \
        --overwrite

Raw Wavelet timestamps describe packet arrival and raw Luster timestamps can
be shared by several frames.  The conversion therefore anchors uniform 2 kHz
EMG and 120 Hz mocap clocks to each stream's observed start/end timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from scripts.realtime.filter import filter_emg_fft


LEFT_EMG_PERMUTATION = np.asarray([1, 0, 7, 6, 5, 4, 3, 2], dtype=np.int64)
RIGHT_EMG_PERMUTATION = np.arange(8, dtype=np.int64)
# Wavelet HDF5 stores integer-valued nanovolts.  EgoEMG memmaps store EMG in
# microvolts, matching the approximately tens-of-uV raw magnitude of the
# existing corpus.
EMG_NANOVOLTS_TO_MICROVOLTS = np.float32(1e-3)
EXPECTED_MARKER_NAMES = (
    "wrist",
    "thumb1",
    "thumb2",
    "thumb3",
    "thumb4",
    "index1",
    "index2",
    "index3",
    "index4",
    "middle1",
    "middle2",
    "middle3",
    "middle4",
    "ring1",
    "ring2",
    "ring3",
    "ring4",
    "pinky1",
    "pinky2",
    "pinky3",
    "pinky4",
)


@dataclass(frozen=True)
class EpisodePlan:
    source_dir: Path
    relative_path: str
    episode_id: str
    subject: str
    subject_id: int
    task_id: str
    task_index: int
    session_type: str
    start_us: int
    end_us: int
    length: int
    left_length: int
    right_length: int
    mocap_length: int


def _decode_names(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def _uniform_clock(start_us: int, end_us: int, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([start_us], dtype=np.int64)
    return np.rint(np.linspace(start_us, end_us, count)).astype(np.int64)


def _nearest_indices(source_us: np.ndarray, target_us: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_us, target_us, side="left")
    right = np.clip(right, 0, len(source_us) - 1)
    left = np.clip(right - 1, 0, len(source_us) - 1)
    choose_left = np.abs(target_us - source_us[left]) <= np.abs(
        source_us[right] - target_us
    )
    return np.where(choose_left, left, right).astype(np.int64)


def _read_video_timestamps_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"Unexpected video timestamp format: {path}")
    # Column 1 is the capture timestamp; column 2 is arrival/processing time.
    values = data[:, 1]
    return values // 1000 if np.median(values) > 10**17 else values


def _read_zed_timestamps(path: Path) -> np.ndarray:
    rows: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(int(json.loads(line)["timestamp"]))
    return np.asarray(rows, dtype=np.int64)


def _video_info(
    task_dir: Path, stream_dir: str
) -> tuple[str, np.ndarray] | None:
    directory = task_dir / stream_dir
    video_paths = sorted(directory.glob("*.mkv"))
    timestamp_paths = sorted(directory.glob("*.txt"))
    if not video_paths and not timestamp_paths:
        return None
    if len(video_paths) != 1 or len(timestamp_paths) != 1:
        raise ValueError(f"Expected one MKV and one TXT under {directory}")
    return str(video_paths[0]), _read_video_timestamps_txt(timestamp_paths[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _soft_fft_filter(data: np.ndarray, sample_rate: float = 2000.0) -> np.ndarray:
    """Apply the shared canonical ``filtered_paper`` implementation."""
    return filter_emg_fft(data, fs=sample_rate)


def _find_task_dirs(source_root: Path) -> list[Path]:
    return sorted(
        metadata.parent
        for metadata in source_root.glob("20??????_*/*/metadata.json")
    )


def _resolve_task_dirs(source_root: Path, requested: list[str] | None) -> list[Path]:
    if not requested:
        return _find_task_dirs(source_root)
    result = [source_root / item for item in requested]
    for task_dir in result:
        if not (task_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"Not a ShowEE task directory: {task_dir}")
    return result


def _inspect_episode(
    source_root: Path,
    task_dir: Path,
    episode_idx: int,
    subject_to_id: dict[str, int],
) -> EpisodePlan:
    metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    subject = str(metadata["session"]["subject_id"])
    subject_id = subject_to_id.setdefault(subject, len(subject_to_id))
    left_path = task_dir / "wavelet_left_wrist" / "wavelet.h5"
    right_path = task_dir / "wavelet_right_wrist" / "wavelet.h5"
    mocap_path = task_dir / "luster_mocap" / "mocap.h5"
    for path in (left_path, right_path, mocap_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with h5py.File(left_path, "r") as left_h5, h5py.File(
        right_path, "r"
    ) as right_h5, h5py.File(mocap_path, "r") as mocap_h5:
        left_ts = left_h5["emg/timestamp"][:]
        right_ts = right_h5["emg/timestamp"][:]
        mocap_ts = mocap_h5["timestamp"][:]
        for hand in ("left_hand", "right_hand"):
            marker_names = _decode_names(mocap_h5[f"{hand}/marker_names"][:])
            if marker_names != EXPECTED_MARKER_NAMES:
                raise ValueError(
                    f"Unexpected marker order in {mocap_path}:{hand}: {marker_names}"
                )

    start_us = max(int(left_ts[0]), int(right_ts[0]))
    end_us = min(int(left_ts[-1]), int(right_ts[-1]))
    if end_us <= start_us:
        raise ValueError(f"No common bilateral EMG time range: {task_dir}")
    length = int(round((end_us - start_us) * 2000.0 / 1_000_000.0)) + 1
    relative_path = task_dir.relative_to(source_root).as_posix()
    return EpisodePlan(
        source_dir=task_dir,
        relative_path=relative_path,
        episode_id=f"episode_{episode_idx:06d}",
        subject=subject,
        subject_id=subject_id,
        task_id=str(metadata["task"]["id"]),
        task_index=int(metadata["task"]["index"]),
        session_type=str(metadata["session"]["session_type"]),
        start_us=start_us,
        end_us=end_us,
        length=length,
        left_length=len(left_ts),
        right_length=len(right_ts),
        mocap_length=len(mocap_ts),
    )


def _field(
    root: Path,
    manifest: dict[str, Any],
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> np.memmap:
    filename = f"{name}.dat"
    manifest["fields"][name] = {
        "filename": filename,
        "dtype": dtype,
        "shape": list(shape),
    }
    return np.memmap(root / filename, mode="w+", dtype=dtype, shape=shape)


def _episode_field(
    root: Path,
    manifest: dict[str, Any],
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> np.memmap:
    filename = f"{name}.dat"
    manifest["episode_fields"][name] = {
        "filename": filename,
        "dtype": dtype,
        "shape": list(shape),
    }
    return np.memmap(root / filename, mode="w+", dtype=dtype, shape=shape)


def _iter_memmaps(value: Any) -> Iterable[np.memmap]:
    if isinstance(value, np.memmap):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_memmaps(item)


def _write_episode(
    plan: EpisodePlan,
    target_us: np.ndarray,
    frame: dict[str, np.memmap],
    destination: slice,
) -> dict[str, Any]:
    task_dir = plan.source_dir
    with h5py.File(
        task_dir / "wavelet_left_wrist" / "wavelet.h5", "r"
    ) as left_h5, h5py.File(
        task_dir / "wavelet_right_wrist" / "wavelet.h5", "r"
    ) as right_h5:
        left_raw = (
            left_h5["emg/data"][:].astype(np.float32)
            * EMG_NANOVOLTS_TO_MICROVOLTS
        )
        right_raw = (
            right_h5["emg/data"][:].astype(np.float32)
            * EMG_NANOVOLTS_TO_MICROVOLTS
        )
        left_clock = _uniform_clock(plan.start_us, plan.end_us, len(left_raw))
        right_clock = _uniform_clock(plan.start_us, plan.end_us, len(right_raw))
        left_idx = _nearest_indices(left_clock, target_us)
        right_idx = _nearest_indices(right_clock, target_us)
        left = left_raw[left_idx][:, LEFT_EMG_PERMUTATION]
        right = right_raw[right_idx][:, RIGHT_EMG_PERMUTATION]
        frame["emg_left_raw"][destination] = left
        frame["emg_right_raw"][destination] = right
        frame["emg_left_filtered_paper"][destination] = _soft_fft_filter(left)
        frame["emg_right_filtered_paper"][destination] = _soft_fft_filter(right)

        imu_acc = left_h5["imu/acc"][:].astype(np.float32)
        imu_gyro = left_h5["imu/gyro"][:].astype(np.float32)
        imu_timestamp = left_h5["imu/timestamp"][:].astype(np.int64)
        imu_clock = _uniform_clock(
            int(imu_timestamp[0]), int(imu_timestamp[-1]), len(imu_timestamp)
        )
        imu_idx = _nearest_indices(imu_clock, target_us)
        frame["imu"][destination] = np.concatenate(
            [imu_acc[imu_idx], imu_gyro[imu_idx]], axis=1
        )

    with h5py.File(task_dir / "luster_mocap" / "mocap.h5", "r") as mocap_h5:
        mocap_timestamp = mocap_h5["timestamp"][:].astype(np.int64)
        mocap_clock = _uniform_clock(
            int(mocap_timestamp[0]), int(mocap_timestamp[-1]), len(mocap_timestamp)
        )
        mocap_idx = _nearest_indices(mocap_clock, target_us)
        for hand in ("left", "right"):
            # Luster stores world-space positions in millimetres; EgoEMG
            # memmaps and markers2mano use metres.
            markers = (
                mocap_h5[f"{hand}_hand/markers"][:].astype(np.float32) / 1000.0
            )
            joints = mocap_h5[f"{hand}_hand/joints"][:].astype(np.float32)
            finite = np.isfinite(markers).all(axis=2)
            valid = (joints[..., 3] > 0.5) & finite
            markers = np.nan_to_num(markers, nan=0.0, posinf=0.0, neginf=0.0)
            frame[f"mocap_{hand}_keypoints"][destination] = markers[mocap_idx]
            frame[f"mocap_{hand}_valid"][destination] = valid[mocap_idx]
            frame[f"mocap_{hand}_wrist_position"][destination] = (
                mocap_h5[f"{hand}_wrist/position"][:][mocap_idx] / 1000.0
            )
            frame[f"mocap_{hand}_wrist_orientation"][destination] = mocap_h5[
                f"{hand}_wrist/quaternion"
            ][:][mocap_idx]
        frame["mocap_head_position"][destination] = (
            mocap_h5["head/position"][:][mocap_idx] / 1000.0
        )
        frame["mocap_head_orientation"][destination] = mocap_h5[
            "head/quaternion"
        ][:][mocap_idx]
        frame["mocap_head_tracked"][destination] = mocap_h5["head/is_tracked"][:][
            mocap_idx
        ]

    webcam_info = _video_info(task_dir, "showee_head")
    zed_path = task_dir / "zed_rgbd" / "rgb.mkv"
    zed_us = _read_zed_timestamps(task_dir / "zed_rgbd" / "rgb_timestamps.jsonl")
    zed_idx = _nearest_indices(zed_us, target_us)
    zed_delta = np.rint((zed_us[zed_idx] - target_us) / 1000.0).astype(np.int32)
    if webcam_info is None:
        webcam_path = ""
        frame["image_head_frame_index"][destination] = -1
        frame["image_head_delta_ms"][destination] = np.iinfo(np.int32).max
        frame["image_head_stale"][destination] = True
    else:
        webcam_path, webcam_us = webcam_info
        webcam_idx = _nearest_indices(webcam_us, target_us)
        webcam_delta = np.rint(
            (webcam_us[webcam_idx] - target_us) / 1000.0
        ).astype(np.int32)
        frame["image_head_frame_index"][destination] = webcam_idx
        frame["image_head_delta_ms"][destination] = webcam_delta
        frame["image_head_stale"][destination] = np.abs(webcam_delta) > 50
    frame["image_zed_frame_index"][destination] = zed_idx
    frame["image_zed_delta_ms"][destination] = zed_delta
    frame["image_zed_stale"][destination] = np.abs(zed_delta) > 50
    return {
        "webcam_path": webcam_path,
        "zed_path": str(zed_path),
        "left_source": left_raw,
        "left_source_indices": left_idx,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--episode", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--markers2mano-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Keep the logical source path: data/showee contains per-session symlinks,
    # and resolving individual children would make them unrelated filesystem
    # paths even though they share this dataset root.
    source_root = args.source_root.absolute()
    out_root = args.out_root.resolve()
    task_dirs = _resolve_task_dirs(source_root, args.episode)
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists (pass --overwrite): {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    subject_to_id: dict[str, int] = {}
    plans = [
        _inspect_episode(source_root, task_dir, idx, subject_to_id)
        for idx, task_dir in enumerate(task_dirs)
    ]
    total_rows = sum(plan.length for plan in plans)
    manifest: dict[str, Any] = {
        "format_version": "egoemg_v2_memmap",
        "total_rows": total_rows,
        "num_episodes": len(plans),
        "source_root": str(source_root),
        "source_format": "showee_raw_hdf5_v1",
        "left_hand_strategy": "flip_local_z",
        "mano_label_policy": "markers2mano_v51_canonical_mano_right_both_hands",
        "emg_channel_layout": {
            "permutation_semantics": (
                "target_channel_i = source_channel[permutation[i]]"
            ),
            "left_source_to_canonical": LEFT_EMG_PERMUTATION.tolist(),
            "right_source_to_canonical": RIGHT_EMG_PERMUTATION.tolist(),
        },
        "emg_units": {
            "source": "nanovolts (integer-valued Wavelet HDF5 samples)",
            "target": "microvolts",
            "scale": float(EMG_NANOVOLTS_TO_MICROVOLTS),
        },
        "clock_reconstruction": {
            "emg_sample_rate_hz": 2000,
            "mocap_sample_rate_hz": 120,
            "policy": "uniform clocks anchored to observed stream start/end timestamps",
            "reason": "raw timestamps are packet arrival timestamps",
        },
        "mocap_position_units": "metres (raw Luster millimetres divided by 1000)",
        "fields": {},
        "episode_fields": {},
    }
    if args.markers2mano_checkpoint is not None:
        checkpoint = args.markers2mano_checkpoint.resolve()
        manifest["markers2mano"] = {
            "checkpoint": str(checkpoint),
            "sha256": _sha256(checkpoint),
        }

    n = total_rows
    e = len(plans)
    specs = {
        "timestamp": ("float64", (n,)),
        "timestamp_us": ("int64", (n,)),
        "episode_index": ("int64", (n,)),
        "frame_index": ("int64", (n,)),
        "source_index": ("int64", (n,)),
        "task_index": ("int64", (n,)),
        "subject_id": ("int32", (n,)),
        "is_first": ("bool", (n,)),
        "is_last": ("bool", (n,)),
        "is_terminal": ("bool", (n,)),
        "label_gesture_class": ("int32", (n,)),
        "label_gesture_active": ("bool", (n,)),
        "emg_left_raw": ("float32", (n, 8)),
        "emg_right_raw": ("float32", (n, 8)),
        "emg_left_filtered_paper": ("float32", (n, 8)),
        "emg_right_filtered_paper": ("float32", (n, 8)),
        "imu": ("float32", (n, 6)),
        "mocap_left_keypoints": ("float32", (n, 21, 3)),
        "mocap_right_keypoints": ("float32", (n, 21, 3)),
        "mocap_left_valid": ("bool", (n, 21)),
        "mocap_right_valid": ("bool", (n, 21)),
        "mocap_left_wrist_position": ("float32", (n, 3)),
        "mocap_left_wrist_orientation": ("float32", (n, 4)),
        "mocap_right_wrist_position": ("float32", (n, 3)),
        "mocap_right_wrist_orientation": ("float32", (n, 4)),
        "mocap_head_position": ("float32", (n, 3)),
        "mocap_head_orientation": ("float32", (n, 4)),
        "mocap_head_tracked": ("bool", (n,)),
        "image_zed_frame_index": ("int32", (n,)),
        "image_head_frame_index": ("int32", (n,)),
        "image_zed_stale": ("bool", (n,)),
        "image_zed_delta_ms": ("int32", (n,)),
        "image_head_stale": ("bool", (n,)),
        "image_head_delta_ms": ("int32", (n,)),
        "generated_mano_left_pose": ("float32", (n, 48)),
        "generated_mano_right_pose": ("float32", (n, 48)),
        "generated_label_valid": ("bool", (n, 2)),
    }
    frame = {
        name: _field(out_root, manifest, name, dtype, shape)
        for name, (dtype, shape) in specs.items()
    }
    episode_fields = {
        "generated_mano_left_beta": _episode_field(
            out_root, manifest, "generated_mano_left_beta", "float32", (e, 10)
        ),
        "generated_mano_right_beta": _episode_field(
            out_root, manifest, "generated_mano_right_beta", "float32", (e, 10)
        ),
    }

    starts: list[int] = []
    ends: list[int] = []
    webcam_paths: list[str] = []
    zed_paths: list[str] = []
    permutation_checks: list[dict[str, Any]] = []
    cursor = 0
    for episode_idx, plan in enumerate(plans):
        start = cursor
        end = start + plan.length
        destination = slice(start, end)
        target_us = _uniform_clock(plan.start_us, plan.end_us, plan.length)
        frame["timestamp_us"][destination] = target_us
        frame["timestamp"][destination] = target_us / 1_000_000.0
        frame["episode_index"][destination] = episode_idx
        frame["frame_index"][destination] = np.arange(plan.length)
        frame["source_index"][destination] = np.arange(plan.length)
        frame["task_index"][destination] = plan.task_index
        frame["subject_id"][destination] = plan.subject_id
        frame["label_gesture_class"][destination] = plan.task_index
        frame["label_gesture_active"][destination] = True
        frame["is_first"][start] = True
        frame["is_last"][end - 1] = True
        frame["is_terminal"][end - 1] = True
        info = _write_episode(plan, target_us, frame, destination)
        expected = info["left_source"][info["left_source_indices"]][
            :, LEFT_EMG_PERMUTATION
        ]
        actual = np.asarray(frame["emg_left_raw"][destination])
        exact = bool(np.array_equal(actual, expected))
        if not exact:
            raise AssertionError(
                f"Left EMG permutation check failed: {plan.relative_path}"
            )
        permutation_checks.append(
            {
                "episode": plan.relative_path,
                "exact_match": exact,
                "samples_checked": plan.length,
                "permutation": LEFT_EMG_PERMUTATION.tolist(),
            }
        )
        starts.append(start)
        ends.append(end)
        webcam_paths.append(info["webcam_path"])
        zed_paths.append(info["zed_path"])
        cursor = end
        print(f"[{episode_idx + 1}/{e}] wrote {plan.relative_path}: {plan.length} rows")

    for memmap in _iter_memmaps(frame):
        memmap.flush()
    for memmap in _iter_memmaps(episode_fields):
        memmap.flush()

    def strings(values: list[str]) -> np.ndarray:
        width = max(1, max(len(value.encode("utf-8")) for value in values))
        return np.asarray(
            [value.encode("utf-8") for value in values], dtype=f"S{width}"
        )

    subjects = [
        name
        for name, _ in sorted(subject_to_id.items(), key=lambda item: item[1])
    ]
    np.savez(
        out_root / "metadata.npz",
        episode_id=strings([plan.episode_id for plan in plans]),
        episode_chunk_id=strings(["chunk-000"] * e),
        episode_subject=strings([plan.subject for plan in plans]),
        episode_subject_id=np.asarray(
            [plan.subject_id for plan in plans], dtype=np.int32
        ),
        episode_source_parquet=strings([plan.relative_path for plan in plans]),
        episode_zed_video_path=strings(zed_paths),
        episode_head_video_path=strings(webcam_paths),
        episode_start_idx=np.asarray(starts, dtype=np.int64),
        episode_end_idx=np.asarray(ends, dtype=np.int64),
        episode_length=np.asarray([plan.length for plan in plans], dtype=np.int64),
        episode_beta_idx=np.arange(e, dtype=np.int32),
        episode_split_id=np.zeros(e, dtype=np.int32),
        subjects_subject=strings(subjects),
        subjects_subject_id=np.arange(len(subjects), dtype=np.int32),
        splits_split=strings(["train"]),
        splits_split_id=np.asarray([0], dtype=np.int32),
    )
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "format_version": manifest["format_version"],
        "source_root": str(source_root),
        "episodes": e,
        "total_rows": total_rows,
        "left_emg_permutation": LEFT_EMG_PERMUTATION.tolist(),
        "left_emg_permutation_checks": permutation_checks,
        "episodes_detail": [
            {
                "episode_id": plan.episode_id,
                "source": plan.relative_path,
                "subject": plan.subject,
                "session_type": plan.session_type,
                "task": plan.task_id,
                "rows": plan.length,
                "raw_left_rows": plan.left_length,
                "raw_right_rows": plan.right_length,
                "raw_mocap_rows": plan.mocap_length,
            }
            for plan in plans
        ],
    }
    (out_root / "metadata_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Done: {out_root} ({total_rows} rows, {e} episodes)")


if __name__ == "__main__":
    main()
