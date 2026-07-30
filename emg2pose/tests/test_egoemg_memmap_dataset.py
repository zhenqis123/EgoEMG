"""Smoke test for EgoEmgMemmapDataset main-path __getitem__ + dataset_source_id.

Regression guard: the per-sample dataset_name restoration (for physically-merged
memmaps) references `center_idx` in the main __getitem__ path, which must be
bound. This builds a tiny synthetic memmap (with the new dataset_source_id
field) and exercises both the main path and _resolve_dataset_name.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _build_synthetic_memmap(out_dir: Path, n_rows: int = 4000, n_episodes: int = 2):
    """Write a minimal EgoEMG-v2-format memmap with dataset_source_id."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wl = 1000  # short windows for a fast test

    fields = {
        "emg_left_raw": ("float32", (n_rows, 8)),
        "emg_left_filtered_paper": ("float32", (n_rows, 8)),
        "emg_right_raw": ("float32", (n_rows, 8)),
        "emg_right_filtered_paper": ("float32", (n_rows, 8)),
        "generated_joint_angles_left": ("float32", (n_rows, 20)),
        "generated_joint_angles_right": ("float32", (n_rows, 20)),
        "generated_label_valid": ("bool", (n_rows, 2)),
        "mocap_left_wrist_pitch": ("float32", (n_rows,)),
        "mocap_left_wrist_yaw": ("float32", (n_rows,)),
        "mocap_right_wrist_pitch": ("float32", (n_rows,)),
        "mocap_right_wrist_yaw": ("float32", (n_rows,)),
        "frame_split_id": ("int8", (n_rows,)),
        "episode_index": ("int64", (n_rows,)),
        "subject_id": ("int32", (n_rows,)),
        "timestamp": ("float64", (n_rows,)),
        "timestamp_us": ("int64", (n_rows,)),
        "dataset_source_id": ("int8", (n_rows,)),
        "is_first": ("bool", (n_rows,)),
        "is_last": ("bool", (n_rows,)),
        "is_terminal": ("bool", (n_rows,)),
        "frame_index": ("int64", (n_rows,)),
        "source_index": ("int64", (n_rows,)),
        "task_index": ("int64", (n_rows,)),
    }
    episode_fields = {
        "generated_mano_left_beta": ("float32", (n_episodes, 10)),
        "generated_mano_right_beta": ("float32", (n_episodes, 10)),
    }

    manifest_fields = {}
    for name, (dtype, shape) in fields.items():
        arr = np.memmap(out_dir / f"{name}.dat", dtype=dtype, mode="w+", shape=shape)
        # Fill with small nonzero values so normalization/stats are finite.
        if dtype == "bool":
            arr[:] = True
        elif name == "dataset_source_id":
            arr[:] = 0  # all EgoEMG source
        elif name == "frame_split_id":
            arr[:] = 0  # all train
        else:
            arr[:] = np.random.RandomState(0).randn(*shape).astype(dtype) * 0.1
        arr.flush()
        manifest_fields[name] = {
            "filename": f"{name}.dat",
            "dtype": dtype,
            "shape": list(shape),
        }
    manifest_ep_fields = {}
    for name, (dtype, shape) in episode_fields.items():
        arr = np.memmap(out_dir / f"{name}.dat", dtype=dtype, mode="w+", shape=shape)
        arr[:] = 0.0
        arr.flush()
        manifest_ep_fields[name] = {
            "filename": f"{name}.dat",
            "dtype": dtype,
            "shape": list(shape),
        }

    # episode_index: split rows evenly across episodes.
    ep_idx = np.memmap(out_dir / "episode_index.dat", dtype="int64", mode="r+", shape=(n_rows,))
    per = n_rows // n_episodes
    for e in range(n_episodes):
        ep_idx[e * per:(e + 1) * per] = e
    ep_idx.flush()

    manifest = {
        "format_version": "egoemg_v2_memmap",
        "total_rows": n_rows,
        "num_episodes": n_episodes,
        "fields": manifest_fields,
        "episode_fields": manifest_ep_fields,
        "generated_joint_angles_semantics": [f"joint_{i}" for i in range(20)],
        "frame_split_labels": ["train", "user", "gesture", "both"],
        "dataset_sources": {"0": "egoemg", "1": "showee", "2": "egoemg_incre"},
    }
    json.dump(manifest, open(out_dir / "manifest.json", "w"))

    metadata = {
        "episode_id": np.array([f"ep_{e}".encode() for e in range(n_episodes)]),
        "episode_subject": np.array([f"subj_{e}".encode() for e in range(n_episodes)]),
        "episode_subject_id": np.arange(n_episodes, dtype=np.int32),
        "episode_source_parquet": np.array([b""] * n_episodes),
        "episode_zed_video_path": np.array([b""] * n_episodes),
        "episode_webcam_video_path": np.array([b""] * n_episodes),
        "episode_start_idx": np.array([e * per for e in range(n_episodes)], dtype=np.int64),
        "episode_end_idx": np.array([(e + 1) * per - 1 for e in range(n_episodes)], dtype=np.int64),
        "episode_length": np.array([per] * n_episodes, dtype=np.int64),
        "episode_beta_idx": np.arange(n_episodes, dtype=np.int32),
        "episode_split_id": np.zeros(n_episodes, dtype=np.int32),
        "episode_chunk_id": np.array([f"ep_{e}".encode() for e in range(n_episodes)]),
        "subjects_subject": np.array([f"subj_{e}".encode() for e in range(n_episodes)]),
        "subjects_subject_id": np.arange(n_episodes, dtype=np.int32),
        "splits_split": np.array([b"train", b"user", b"gesture", b"both"]),
        "splits_split_id": np.array([0, 1, 2, 3], dtype=np.int32),
    }
    np.savez(out_dir / "metadata.npz", **metadata)
    return out_dir


@pytest.mark.parametrize("hand", ["left", "right"])
def test_dataset_main_path_getitem_and_dataset_name(tmp_path, hand):
    """Main __getitem__ path runs without NameError; dataset_name resolves."""
    from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

    memmap_dir = _build_synthetic_memmap(tmp_path / "synth_memmap")
    ds = EgoEmgMemmapDataset(
        memmap_dir=memmap_dir,
        window_length=1000,
        stride=1000,
        allowed_splits=["train"],
        modalities=["emg", "joint_angles", "labels"],
        target_hand=hand,
        emg_field_preference="filtered_paper",
        emg_layout="target_hand",
        emg2pose_channel_indices=None,
        channel_interpolate=False,
        norm_mode=None,
        norm_stats_path=None,
        dataset_name="egoemg_unified",
        jitter=False,
        center_target_only=False,
    )
    assert len(ds) > 0, "dataset should have at least one window"
    # Main (full-window) path — this is the path that had the NameError.
    sample = ds[0]
    assert "emg" in sample, "sample must contain EMG"
    assert "dataset_name" in sample, "sample must expose dataset_name"
    assert sample["dataset_name"] == "egoemg", (
        f"per-sample dataset_name should resolve to 'egoemg' via dataset_source_id, "
        f"got {sample['dataset_name']!r}"
    )


def test_resolve_dataset_name_fallback_no_source_field(tmp_path):
    """When dataset_source_id is absent, fall back to instance dataset_name."""
    from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

    memmap_dir = _build_synthetic_memmap(tmp_path / "synth_memmap_no_src")
    # Remove the dataset_source_id field + manifest entries to simulate a legacy
    # (non-merged) memmap with no source provenance.
    (memmap_dir / "dataset_source_id.dat").unlink()
    manifest = json.load(open(memmap_dir / "manifest.json"))
    del manifest["fields"]["dataset_source_id"]
    manifest.pop("dataset_sources", None)
    json.dump(manifest, open(memmap_dir / "manifest.json", "w"))

    ds = EgoEmgMemmapDataset(
        memmap_dir=memmap_dir,
        window_length=1000, stride=1000, allowed_splits=["train"],
        modalities=["emg", "joint_angles", "labels"], target_hand="right",
        emg_field_preference="filtered_paper", emg_layout="target_hand",
        emg2pose_channel_indices=None, channel_interpolate=False,
        norm_mode=None, norm_stats_path=None, dataset_name="legacy_name",
        jitter=False, center_target_only=False,
    )
    ds._open_memmaps()
    assert ds._source_id_to_name is None, "no dataset_sources -> map is None"
    # Fallback path returns the instance dataset_name regardless of center.
    assert ds._resolve_dataset_name(center_idx=0) == "legacy_name"
