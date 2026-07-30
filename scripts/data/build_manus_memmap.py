#!/usr/bin/env python3
"""Build EgoEMG-compatible memmap from fitted Manus sessions.

Reads sessions from data/manus_fit/ (produced by fit_manus_to_umetrack.py) and
data/data/ (raw EMG), then writes a memmap dataset that EgoEmgMemmapDataset can
load directly for EMGFormer training.

EMG preprocessing (matching filter_emg_into_new_columns.py):
  1. Convert µV → mV (raw Manus EMG is in µV, EgoEMG uses mV)
  2. FFT-domain filter: notch 50/100 Hz + bandpass 20-850 Hz
  3. emg_right_raw = µV→mV only; emg_right_filtered = µV→mV + FFT filter

Usage:
  python scripts/data/build_manus_memmap.py \
    --data-root data/data \
    --fit-root data/manus_fit \
    --output data/manus_memmap
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ── EMG filter constants (matching filter_emg_into_new_columns.py) ──
FS = 2000.0
LOW_CUT = 20.0
LOW_TRANSITION = 5.0
HIGH_CUT = 850.0
HIGH_TRANSITION = 50.0
NOTCH_CONFIGS = (
    {"center": 50.0, "stop_half_width": 1.5, "transition_half_width": 1.5},
    {"center": 100.0, "stop_half_width": 1.5, "transition_half_width": 1.5},
)


def _smoothstep_cosine(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def _build_frequency_mask(n_samples: int, fs: float = FS) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = np.ones_like(freqs, dtype=np.float64)

    # High-pass with soft roll-off
    hp0 = max(0.0, LOW_CUT - LOW_TRANSITION)
    hp1 = LOW_CUT
    if hp1 > hp0:
        below = freqs <= hp0
        trans = (freqs > hp0) & (freqs < hp1)
        mask[below] = 0.0
        mask[trans] *= _smoothstep_cosine((freqs[trans] - hp0) / (hp1 - hp0))

    # Low-pass with soft roll-off
    lp0 = HIGH_CUT
    lp1 = min(fs * 0.5, HIGH_CUT + HIGH_TRANSITION)
    if lp1 > lp0:
        above = freqs >= lp1
        trans = (freqs > lp0) & (freqs < lp1)
        mask[above] = 0.0
        mask[trans] *= (1.0 - _smoothstep_cosine((freqs[trans] - lp0) / (lp1 - lp0)))

    # Narrow notches for 50 Hz and 100 Hz
    for cfg in NOTCH_CONFIGS:
        center = cfg["center"]
        stop_hw = cfg["stop_half_width"]
        trans_hw = cfg["transition_half_width"]
        stop_lo = center - stop_hw
        stop_hi = center + stop_hw
        trans_lo = stop_lo - trans_hw
        trans_hi = stop_hi + trans_hw

        hard = (freqs >= stop_lo) & (freqs <= stop_hi)
        left = (freqs > trans_lo) & (freqs < stop_lo)
        right = (freqs > stop_hi) & (freqs < trans_hi)
        mask[hard] = 0.0
        if np.any(left):
            mask[left] *= (1.0 - _smoothstep_cosine((freqs[left] - trans_lo) / (stop_lo - trans_lo)))
        if np.any(right):
            mask[right] *= _smoothstep_cosine((freqs[right] - stop_hi) / (trans_hi - stop_hi))

    return mask


def filter_emg_fft(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Apply FFT-domain notch + bandpass filter to EMG.

    x: (N, 8) float32 in mV. Returns (N, 8) float32 filtered.
    """
    if x.size == 0:
        return x.astype(np.float32, copy=True)
    x = x.astype(np.float64, copy=False)
    mean = np.mean(x, axis=0, keepdims=True)
    x0 = x - mean
    mask = _build_frequency_mask(x0.shape[0], fs=fs)[:, None]
    spec = np.fft.rfft(x0, axis=0)
    spec *= mask
    y = np.fft.irfft(spec, n=x0.shape[0], axis=0)
    return y.astype(np.float32)


def load_emg(emg_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load EMG from CSV, convert µV→mV. Returns (timestamps_s, emg_Nx8_mV)."""
    timestamps = []
    emg_rows = []
    with open(emg_csv) as f:
        for row in csv.DictReader(f):
            timestamps.append(float(row["timestamp"]))
            # Raw values are in µV; convert to mV for EgoEMG compatibility
            emg_rows.append([float(row[f"ch{i}"]) / 1000.0 for i in range(1, 9)])
    return np.array(timestamps, dtype=np.float64), np.array(emg_rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build EgoEMG-compatible memmap from fitted Manus sessions"
    )
    parser.add_argument("--data-root", default="data/data")
    parser.add_argument("--fit-root", default="data/manus_fit")
    parser.add_argument("--output", default="data/manus_memmap")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    fit_root = Path(args.fit_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = sorted([
        d.name for d in fit_root.iterdir()
        if d.is_dir() and (data_root / d.name).is_dir()
    ])
    if not sessions:
        print("No sessions found with both raw data and fit output!")
        return
    print(f"Found {len(sessions)} sessions: {sessions}")

    # ── Pass 1: collect per-session arrays and compute global layout ──
    session_data = []
    total_frames = 0

    for session_name in sessions:
        session_data_dir = data_root / session_name
        session_fit_dir = fit_root / session_name

        emg_csv = session_data_dir / "emg.csv"
        if not emg_csv.exists():
            print(f"  SKIP {session_name}: no emg.csv")
            continue

        timestamps, emg_raw_mv = load_emg(emg_csv)
        n_frames = len(timestamps)

        # FFT filter: notch 50/100 Hz + bandpass 20-850 Hz
        print(f"  Filtering EMG ({n_frames} samples × 8ch)...")
        emg_filtered_mv = filter_emg_fft(emg_raw_mv)
        print(f"    raw std: {emg_raw_mv.std(axis=0)}")
        print(f"    filtered std: {emg_filtered_mv.std(axis=0)}")

        # Load fitted data (EMG-aligned)
        angles_path = session_fit_dir / "joint_angles_right_emg_aligned.npy"
        wrist_aa_path = session_fit_dir / "wrist_rotation_right_emg_aligned.npy"
        scales_path = session_fit_dir / "scales_right_emg_aligned.npy"
        trans_path = session_fit_dir / "wrist_translation_right_emg_aligned.npy"

        if not angles_path.exists():
            print(f"  SKIP {session_name}: missing fitted angles (run fit_manus_to_umetrack.py first)")
            continue

        angles = np.load(angles_path)
        wrist_aa = np.load(wrist_aa_path) if wrist_aa_path.exists() else np.zeros((n_frames, 3), dtype=np.float32)
        scales = np.load(scales_path) if scales_path.exists() else np.ones(n_frames, dtype=np.float32)
        trans = np.load(trans_path) if trans_path.exists() else np.zeros((n_frames, 3), dtype=np.float32)

        if scales.ndim == 0:
            scales = np.full(n_frames, float(scales), dtype=np.float32)
        if scales.shape[0] != n_frames:
            scales = np.full(n_frames, float(scales.mean()), dtype=np.float32)

        for name, arr in [("angles", angles), ("wrist_aa", wrist_aa), ("scales", scales), ("trans", trans)]:
            if len(arr) != n_frames:
                print(f"  WARNING {session_name}: {name} length {len(arr)} != EMG {n_frames}")

        # Build timestamps_us
        timestamps_us = (timestamps * 1_000_000).astype(np.int64)

        # Metadata arrays
        frame_index = np.arange(n_frames, dtype=np.int64)
        episode_index = np.full(n_frames, -1, dtype=np.int64)

        # Labels
        generated_label_valid = np.zeros((n_frames, 2), dtype=bool)
        generated_label_valid[:, 1] = True  # right hand valid

        # Left hand: all zeros (no left hand data)
        emg_left = np.zeros_like(emg_raw_mv)

        # Dummy arrays
        zeros_1d = np.zeros(n_frames, dtype=np.float32)
        zeros_bool = np.zeros(n_frames, dtype=bool)
        zeros_i32 = np.full(n_frames, -1, dtype=np.int32)
        zeros_3d = np.zeros((n_frames, 3), dtype=np.float32)
        zeros_4d = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n_frames, 1))
        zeros_6d = np.zeros((n_frames, 6), dtype=np.float32)
        zeros_12d = np.zeros((n_frames, 12), dtype=np.float32)
        zeros_kp = np.zeros((n_frames, 21, 3), dtype=np.float32)
        zeros_kp_valid = np.zeros((n_frames, 21), dtype=bool)
        zeros_l5 = np.zeros((n_frames, 5, 3), dtype=np.float32)
        zeros_l4 = np.zeros((n_frames, 4, 3), dtype=np.float32)
        zeros_mano = np.zeros((n_frames, 48), dtype=np.float32)
        zeros_ja = np.zeros((n_frames, 20), dtype=np.float32)

        is_first = np.zeros(n_frames, dtype=bool)
        is_first[0] = True
        is_last = np.zeros(n_frames, dtype=bool)
        is_last[-1] = True
        is_terminal = np.zeros(n_frames, dtype=bool)
        is_terminal[-1] = True

        session_data.append({
            "session_name": session_name,
            "n_frames": n_frames,
            "start_idx": total_frames,
            "end_idx": total_frames + n_frames - 1,
            # Core fields
            "timestamp": timestamps,
            "timestamp_us": timestamps_us,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "source_index": np.zeros(n_frames, dtype=np.int64),
            "task_index": np.zeros(n_frames, dtype=np.int64),
            "subject_id": np.full(n_frames, len(session_data), dtype=np.int32),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            # EMG: raw mV + FFT filtered mV
            "emg_left_raw": emg_left,
            "emg_right_raw": emg_raw_mv,
            "emg_left_filtered": emg_left,
            "emg_right_filtered": emg_filtered_mv,
            # IMU
            "imu": zeros_6d,
            # Mocap keypoints (dummy)
            "mocap_left_keypoints": zeros_kp.copy(),
            "mocap_right_keypoints": zeros_kp.copy(),
            "mocap_left_valid": zeros_kp_valid.copy(),
            "mocap_right_valid": zeros_kp_valid.copy(),
            "mocap_left_wrist_position": zeros_3d.copy(),
            "mocap_left_wrist_orientation": zeros_4d.copy(),
            "mocap_right_wrist_position": zeros_3d.copy(),
            "mocap_right_wrist_orientation": zeros_4d.copy(),
            "mocap_left_wrist_rigid_markers": zeros_l5.copy(),
            "mocap_right_wrist_rigid_markers": zeros_l4.copy(),
            "mocap_webcam_position": zeros_3d,
            "mocap_webcam_orientation": zeros_4d,
            "mocap_webcam_tracked": zeros_bool,
            "mocap_webcam_rigid_markers": zeros_l4,
            "mocap_webcam_transform": zeros_12d,
            "mocap_mano_left_world_transform": zeros_12d,
            "mocap_mano_right_world_transform": zeros_12d,
            # Labels
            "label_gesture_class": np.zeros(n_frames, dtype=np.int32),
            "label_gesture_active": zeros_bool,
            "generated_label_valid": generated_label_valid,
            # Video/image indices
            "image_zed_frame_index": zeros_i32.copy(),
            "image_webcam_frame_index": zeros_i32.copy(),
            "image_zed_stale": np.ones(n_frames, dtype=bool),
            "image_zed_delta_ms": zeros_i32.copy(),
            "image_webcam_stale": np.ones(n_frames, dtype=bool),
            "image_webcam_delta_ms": zeros_i32.copy(),
            # MANO (dummy)
            "generated_mano_left_pose": zeros_mano.copy(),
            "generated_mano_right_pose": zeros_mano.copy(),
            # Joint angles
            "generated_joint_angles_left": zeros_ja,
            "generated_joint_angles_right": angles,
            # Split
            "frame_split_id": np.zeros(n_frames, dtype=np.int8),
        })
        total_frames += n_frames
        print(f"  {session_name}: {n_frames} EMG frames")

    print(f"\nTotal frames: {total_frames}")

    # ── Pass 2: assign episode_index ──
    for i, sd in enumerate(session_data):
        sd["episode_index"][:] = i

    # ── Pass 3: write memmaps ──
    field_specs = [
        ("timestamp", "timestamp.dat", "float64", (total_frames,)),
        ("timestamp_us", "timestamp_us.dat", "int64", (total_frames,)),
        ("episode_index", "episode_index.dat", "int64", (total_frames,)),
        ("frame_index", "frame_index.dat", "int64", (total_frames,)),
        ("source_index", "source_index.dat", "int64", (total_frames,)),
        ("task_index", "task_index.dat", "int64", (total_frames,)),
        ("subject_id", "subject_id.dat", "int32", (total_frames,)),
        ("is_first", "is_first.dat", "bool", (total_frames,)),
        ("is_last", "is_last.dat", "bool", (total_frames,)),
        ("is_terminal", "is_terminal.dat", "bool", (total_frames,)),
        ("label_gesture_class", "label_gesture_class.dat", "int32", (total_frames,)),
        ("label_gesture_active", "label_gesture_active.dat", "bool", (total_frames,)),
        ("emg_left_raw", "emg_left_raw.dat", "float32", (total_frames, 8)),
        ("emg_right_raw", "emg_right_raw.dat", "float32", (total_frames, 8)),
        ("emg_left_filtered", "emg_left_filtered.dat", "float32", (total_frames, 8)),
        ("emg_right_filtered", "emg_right_filtered.dat", "float32", (total_frames, 8)),
        ("imu", "imu.dat", "float32", (total_frames, 6)),
        ("mocap_left_keypoints", "mocap_left_keypoints.dat", "float32", (total_frames, 21, 3)),
        ("mocap_right_keypoints", "mocap_right_keypoints.dat", "float32", (total_frames, 21, 3)),
        ("mocap_left_valid", "mocap_left_valid.dat", "bool", (total_frames, 21)),
        ("mocap_right_valid", "mocap_right_valid.dat", "bool", (total_frames, 21)),
        ("mocap_left_wrist_position", "mocap_left_wrist_position.dat", "float32", (total_frames, 3)),
        ("mocap_left_wrist_orientation", "mocap_left_wrist_orientation.dat", "float32", (total_frames, 4)),
        ("mocap_right_wrist_position", "mocap_right_wrist_position.dat", "float32", (total_frames, 3)),
        ("mocap_right_wrist_orientation", "mocap_right_wrist_orientation.dat", "float32", (total_frames, 4)),
        ("mocap_left_wrist_rigid_markers", "mocap_left_wrist_rigid_markers.dat", "float32", (total_frames, 5, 3)),
        ("mocap_right_wrist_rigid_markers", "mocap_right_wrist_rigid_markers.dat", "float32", (total_frames, 4, 3)),
        ("mocap_webcam_position", "mocap_webcam_position.dat", "float32", (total_frames, 3)),
        ("mocap_webcam_orientation", "mocap_webcam_orientation.dat", "float32", (total_frames, 4)),
        ("mocap_webcam_tracked", "mocap_webcam_tracked.dat", "bool", (total_frames,)),
        ("mocap_webcam_rigid_markers", "mocap_webcam_rigid_markers.dat", "float32", (total_frames, 4, 3)),
        ("mocap_webcam_transform", "mocap_webcam_transform.dat", "float32", (total_frames, 12)),
        ("mocap_mano_left_world_transform", "mocap_mano_left_world_transform.dat", "float32", (total_frames, 12)),
        ("mocap_mano_right_world_transform", "mocap_mano_right_world_transform.dat", "float32", (total_frames, 12)),
        ("image_zed_frame_index", "image_zed_frame_index.dat", "int32", (total_frames,)),
        ("image_webcam_frame_index", "image_webcam_frame_index.dat", "int32", (total_frames,)),
        ("image_zed_stale", "image_zed_stale.dat", "bool", (total_frames,)),
        ("image_zed_delta_ms", "image_zed_delta_ms.dat", "int32", (total_frames,)),
        ("image_webcam_stale", "image_webcam_stale.dat", "bool", (total_frames,)),
        ("image_webcam_delta_ms", "image_webcam_delta_ms.dat", "int32", (total_frames,)),
        ("generated_mano_left_pose", "generated_mano_left_pose.dat", "float32", (total_frames, 48)),
        ("generated_mano_right_pose", "generated_mano_right_pose.dat", "float32", (total_frames, 48)),
        ("generated_label_valid", "generated_label_valid.dat", "bool", (total_frames, 2)),
        ("generated_joint_angles_left", "generated_joint_angles_left.dat", "float32", (total_frames, 20)),
        ("generated_joint_angles_right", "generated_joint_angles_right.dat", "float32", (total_frames, 20)),
        ("frame_split_id", "frame_split_id.dat", "int8", (total_frames,)),
    ]

    dtype_map = {
        "float32": np.float32, "float64": np.float64,
        "int32": np.int32, "int64": np.int64, "int8": np.int8,
        "bool": bool,
    }

    manifest_fields = {}
    for field_name, filename, dtype_str, shape in field_specs:
        manifest_fields[field_name] = {"filename": filename, "dtype": dtype_str, "shape": list(shape)}

    for field_name, filename, dtype_str, shape in field_specs:
        np_dtype = dtype_map[dtype_str]
        out_path = out_dir / filename
        parts = [sd[field_name] for sd in session_data]
        full = np.concatenate(parts, axis=0) if parts[0].ndim > 0 else np.concatenate([p.reshape(-1) for p in parts])
        full = full.astype(np_dtype)
        full.tofile(out_path)
        print(f"  Wrote {filename}: {full.shape} {full.dtype}")

    # ── Episode fields ──
    num_episodes = len(session_data)
    mano_beta_left = np.zeros((num_episodes, 10), dtype=np.float32)
    mano_beta_right = np.zeros((num_episodes, 10), dtype=np.float32)
    mano_beta_left.tofile(out_dir / "generated_mano_left_beta.dat")
    mano_beta_right.tofile(out_dir / "generated_mano_right_beta.dat")

    # ── Manifest ──
    manifest = {
        "format_version": "egoemg_v2_memmap",
        "total_rows": total_frames,
        "num_episodes": num_episodes,
        "left_hand_strategy": "flip_local_z",
        "mano_label_policy": "generated_only_manus_fit",
        "source_root": str(data_root.resolve()),
        "fields": manifest_fields,
        "episode_fields": {
            "generated_mano_left_beta": {"filename": "generated_mano_left_beta.dat", "dtype": "float32", "shape": [num_episodes, 10]},
            "generated_mano_right_beta": {"filename": "generated_mano_right_beta.dat", "dtype": "float32", "shape": [num_episodes, 10]},
        },
        "generated_joint_angles_semantics": [
            "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe",
            "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
            "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
            "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
            "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
        ],
        "frame_split_labels": ["train"],
        "frame_split_policy": "all_train",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── metadata.npz ──
    episode_ids = [sd["session_name"] for sd in session_data]
    start_indices = np.array([sd["start_idx"] for sd in session_data], dtype=np.int64)
    end_indices = np.array([sd["end_idx"] for sd in session_data], dtype=np.int64)
    lengths = np.array([sd["n_frames"] for sd in session_data], dtype=np.int64)
    beta_indices = np.arange(num_episodes, dtype=np.int32)

    def _str_array(strings: list[str], max_len: int = 256) -> np.ndarray:
        return np.array([s.encode()[:max_len].ljust(max_len, b'\x00') for s in strings], dtype=f'S{max_len}')

    metadata = {
        "episode_id": _str_array(episode_ids, 32),
        "episode_chunk_id": _str_array(["manus" for _ in episode_ids], 16),
        "episode_subject": _str_array(["manus_subject" for _ in episode_ids], 32),
        "episode_subject_id": np.zeros(num_episodes, dtype=np.int32),
        "episode_source_parquet": _str_array(["" for _ in episode_ids], 128),
        "episode_zed_video_path": _str_array(["" for _ in episode_ids], 128),
        "episode_webcam_video_path": _str_array(["" for _ in episode_ids], 128),
        "episode_start_idx": start_indices,
        "episode_end_idx": end_indices,
        "episode_length": lengths,
        "episode_beta_idx": beta_indices,
        "subjects_subject": _str_array(["manus_subject" for _ in episode_ids], 32),
        "subjects_subject_id": np.zeros(num_episodes, dtype=np.int32),
        "splits_split": _str_array(["train"], 8),
        "splits_split_id": np.array([0], dtype=np.int32),
    }
    np.savez(out_dir / "metadata.npz", **metadata)

    # ── Summary ──
    summary = {
        "format_version": "egoemg_v2_memmap",
        "source_root": str(data_root.resolve()),
        "episodes": num_episodes,
        "total_rows": total_frames,
        "left_hand_strategy": "flip_local_z",
        "mano_label_policy": "generated_only_manus_fit",
        "emg_preprocessing": {
            "fs_hz": FS,
            "pipeline": [
                "convert uV to mV (divide by 1000)",
                "subtract per-channel mean",
                "narrow notch around 50 Hz",
                "narrow notch around 100 Hz",
                "wide bandpass 20-850 Hz with soft roll-off",
                "no normalization",
            ],
            "implementation": "fft_frequency_mask",
        },
        "fields": {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in manifest_fields.items()},
        "episode_fields": {
            "generated_mano_left_beta": {"shape": [num_episodes, 10], "dtype": "float32"},
            "generated_mano_right_beta": {"shape": [num_episodes, 10], "dtype": "float32"},
        },
        "generated_joint_angles_semantics": manifest["generated_joint_angles_semantics"],
        "generated_joint_angles_note": "Fitted from Manus glove keypoints via L-BFGS IK optimization (27 params: 20 angles + 3 wrist aa + 1 scale + 3 translation).",
    }
    with open(out_dir / "metadata_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Memmap written to {out_dir}/")
    print(f"  Total frames: {total_frames}")
    print(f"  Sessions: {episode_ids}")


if __name__ == "__main__":
    main()
