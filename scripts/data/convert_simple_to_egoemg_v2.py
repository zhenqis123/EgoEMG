#!/usr/bin/env python3
"""Convert simplified emg2pose memmap to full EgoEMG v2 format.

Reads a simplified memmap (produced by convert_emg2pose_zarr_to_memmap.py)
with fields: emg, emg_filtered, joint_angles, valid_mask, time, timestamp.
Writes a full EgoEMG v2 memmap (42+ fields) compatible with
EgoEmgMemmapDataset for training.

EMG preprocessing (matching build_manus_memmap.py):
  1. Convert µV/ADC → mV (divide by 1000)
  2. FFT-domain filter: notch 50/100 Hz + bandpass 20-850 Hz
  3. emg_right_raw = µV→mV only; emg_right_filtered = re-filtered in mV space

Usage:
  python scripts/data/convert_simple_to_egoemg_v2.py \
    --input data/data_20260526_172725 \
    --output data/data_20260526_172725_egoemg_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── EMG filter constants (matching build_manus_memmap.py) ──
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
        mask[trans] *= 1.0 - _smoothstep_cosine(
            (freqs[trans] - lp0) / (lp1 - lp0)
        )

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
            mask[left] *= 1.0 - _smoothstep_cosine(
                (freqs[left] - trans_lo) / (stop_lo - trans_lo)
            )
        if np.any(right):
            mask[right] *= _smoothstep_cosine(
                (freqs[right] - stop_hi) / (trans_hi - stop_hi)
            )

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


def _str_array(strings: list[str], max_len: int = 256) -> np.ndarray:
    return np.array(
        [s.encode()[:max_len].ljust(max_len, b"\x00") for s in strings],
        dtype=f"S{max_len}",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert simplified memmap to EgoEMG v2 format"
    )
    parser.add_argument(
        "--input",
        default="data/data_20260526_172725",
        help="Input directory with simplified memmap",
    )
    parser.add_argument(
        "--output",
        default="data/data_20260526_172725_egoemg_v2",
        help="Output directory for EgoEMG v2 memmap",
    )
    parser.add_argument(
        "--emg-scale",
        type=float,
        default=1000.0,
        help="Divide raw EMG by this to get mV (default: 1000 for µV→mV)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read source manifest ──
    with open(in_dir / "manifest.json") as f:
        src_manifest = json.load(f)

    src_fields = src_manifest["fields"]
    n_frames = src_manifest["total_rows"]
    session_name = in_dir.name

    print(f"Source: {in_dir}")
    print(f"  Format: {src_manifest['format_version']}")
    print(f"  Total rows: {n_frames}")
    print(f"  Fields: {list(src_fields.keys())}")

    # ── Read source memmaps ──
    print(f"\nReading source data...")

    def _read_field(name: str) -> np.ndarray:
        info = src_fields[name]
        return np.memmap(
            in_dir / info["filename"],
            dtype=np.dtype(info["dtype"]),
            mode="r",
            shape=tuple(info["shape"]),
        )

    emg_raw = _read_field("emg")  # (N, 8) float32, µV/ADC
    joint_angles = _read_field("joint_angles")  # (N, 20) float32
    valid_mask = _read_field("valid_mask")  # (N,) bool
    timestamps = _read_field("timestamp")  # (N,) float64

    print(f"  emg: shape={emg_raw.shape}, mean={emg_raw.mean():.1f}, "
          f"std={emg_raw.std():.1f}")
    print(f"  joint_angles: shape={joint_angles.shape}, "
          f"mean={joint_angles.mean():.4f}, std={joint_angles.std():.4f}")
    print(f"  valid_mask: {valid_mask.sum()}/{len(valid_mask)} valid "
          f"({100 * valid_mask.mean():.1f}%)")

    # ── EMG preprocessing: µV → mV + FFT filter ──
    print(f"\nPreprocessing EMG (÷{args.emg_scale:.0f} → mV, then FFT filter)...")
    emg_mv = np.array(emg_raw, dtype=np.float32) / args.emg_scale
    print(f"  raw mV: mean={emg_mv.mean():.4f}, std={emg_mv.std():.4f}")

    emg_filtered_mv = filter_emg_fft(emg_mv)
    print(f"  filtered mV: mean={emg_filtered_mv.mean():.4f}, "
          f"std={emg_filtered_mv.std():.4f}")
    print(f"  per-channel raw std:    {emg_mv.std(axis=0).round(3).tolist()}")
    print(f"  per-channel filt std:   "
          f"{emg_filtered_mv.std(axis=0).round(4).tolist()}")

    # ── Build all EgoEMG v2 fields ──
    print(f"\nBuilding {n_frames} frames of EgoEMG v2 fields...")

    timestamps_us = (timestamps * 1_000_000).astype(np.int64)
    frame_index = np.arange(n_frames, dtype=np.int64)

    # Label valid: (N, 2) — right hand valid, left hand invalid
    generated_label_valid = np.zeros((n_frames, 2), dtype=bool)
    generated_label_valid[:, 1] = valid_mask  # right hand

    # Left hand: all zeros
    emg_left = np.zeros((n_frames, 8), dtype=np.float32)

    # Boundary flags
    is_first = np.zeros(n_frames, dtype=bool)
    is_first[0] = True
    is_last = np.zeros(n_frames, dtype=bool)
    is_last[-1] = True
    is_terminal = np.zeros(n_frames, dtype=bool)
    is_terminal[-1] = True

    # Dummy arrays
    zeros_f32_1d = np.zeros(n_frames, dtype=np.float32)
    zeros_bool = np.zeros(n_frames, dtype=bool)
    zeros_i32 = np.full(n_frames, -1, dtype=np.int32)
    zeros_3d = np.zeros((n_frames, 3), dtype=np.float32)
    zeros_4d = np.tile(
        np.array([0, 0, 0, 1], dtype=np.float32), (n_frames, 1)
    )
    zeros_6d = np.zeros((n_frames, 6), dtype=np.float32)
    zeros_12d = np.zeros((n_frames, 12), dtype=np.float32)
    zeros_kp = np.zeros((n_frames, 21, 3), dtype=np.float32)
    zeros_kp_valid = np.zeros((n_frames, 21), dtype=bool)
    zeros_l5 = np.zeros((n_frames, 5, 3), dtype=np.float32)
    zeros_l4 = np.zeros((n_frames, 4, 3), dtype=np.float32)
    zeros_mano = np.zeros((n_frames, 48), dtype=np.float32)
    zeros_ja = np.zeros((n_frames, 20), dtype=np.float32)

    # ── Field spec: (name, filename, dtype_str, array) ──
    field_data = {
        "timestamp": ("float64", (n_frames,), np.array(timestamps, dtype=np.float64)),
        "timestamp_us": ("int64", (n_frames,), timestamps_us),
        "episode_index": ("int64", (n_frames,), np.zeros(n_frames, dtype=np.int64)),
        "frame_index": ("int64", (n_frames,), frame_index),
        "source_index": ("int64", (n_frames,), np.zeros(n_frames, dtype=np.int64)),
        "task_index": ("int64", (n_frames,), np.zeros(n_frames, dtype=np.int64)),
        "subject_id": ("int32", (n_frames,), np.zeros(n_frames, dtype=np.int32)),
        "is_first": ("bool", (n_frames,), is_first),
        "is_last": ("bool", (n_frames,), is_last),
        "is_terminal": ("bool", (n_frames,), is_terminal),
        # Labels
        "label_gesture_class": ("int32", (n_frames,), np.zeros(n_frames, dtype=np.int32)),
        "label_gesture_active": ("bool", (n_frames,), zeros_bool.copy()),
        "generated_label_valid": ("bool", (n_frames, 2), generated_label_valid),
        # EMG
        "emg_left_raw": ("float32", (n_frames, 8), emg_left.copy()),
        "emg_right_raw": ("float32", (n_frames, 8), emg_mv),
        "emg_left_filtered": ("float32", (n_frames, 8), emg_left.copy()),
        "emg_right_filtered": ("float32", (n_frames, 8), emg_filtered_mv),
        # IMU
        "imu": ("float32", (n_frames, 6), zeros_6d),
        # Mocap keypoints
        "mocap_left_keypoints": ("float32", (n_frames, 21, 3), zeros_kp.copy()),
        "mocap_right_keypoints": ("float32", (n_frames, 21, 3), zeros_kp.copy()),
        "mocap_left_valid": ("bool", (n_frames, 21), zeros_kp_valid.copy()),
        "mocap_right_valid": ("bool", (n_frames, 21), zeros_kp_valid.copy()),
        # Wrist
        "mocap_left_wrist_position": ("float32", (n_frames, 3), zeros_3d.copy()),
        "mocap_left_wrist_orientation": ("float32", (n_frames, 4), zeros_4d.copy()),
        "mocap_left_wrist_pitch": ("float32", (n_frames,), zeros_f32_1d.copy()),
        "mocap_left_wrist_yaw": ("float32", (n_frames,), zeros_f32_1d.copy()),
        "mocap_left_wrist_angles_valid": ("bool", (n_frames,), zeros_bool.copy()),
        "mocap_left_wrist_rigid_markers": ("float32", (n_frames, 5, 3), zeros_l5.copy()),
        "mocap_right_wrist_position": ("float32", (n_frames, 3), zeros_3d.copy()),
        "mocap_right_wrist_orientation": ("float32", (n_frames, 4), zeros_4d.copy()),
        "mocap_right_wrist_pitch": ("float32", (n_frames,), zeros_f32_1d.copy()),
        "mocap_right_wrist_yaw": ("float32", (n_frames,), zeros_f32_1d.copy()),
        "mocap_right_wrist_angles_valid": ("bool", (n_frames,), zeros_bool.copy()),
        "mocap_right_wrist_rigid_markers": ("float32", (n_frames, 4, 3), zeros_l4.copy()),
        # Webcam mocap
        "mocap_webcam_position": ("float32", (n_frames, 3), zeros_3d.copy()),
        "mocap_webcam_orientation": ("float32", (n_frames, 4), zeros_4d.copy()),
        "mocap_webcam_tracked": ("bool", (n_frames,), zeros_bool.copy()),
        "mocap_webcam_rigid_markers": ("float32", (n_frames, 4, 3), zeros_l4.copy()),
        "mocap_webcam_transform": ("float32", (n_frames, 12), zeros_12d.copy()),
        # MANO world transforms
        "mocap_mano_left_world_transform": ("float32", (n_frames, 12), zeros_12d.copy()),
        "mocap_mano_right_world_transform": ("float32", (n_frames, 12), zeros_12d.copy()),
        # Video/image indices
        "image_zed_frame_index": ("int32", (n_frames,), zeros_i32.copy()),
        "image_webcam_frame_index": ("int32", (n_frames,), zeros_i32.copy()),
        "image_zed_stale": ("bool", (n_frames,), np.ones(n_frames, dtype=bool)),
        "image_zed_delta_ms": ("int32", (n_frames,), zeros_i32.copy()),
        "image_webcam_stale": ("bool", (n_frames,), np.ones(n_frames, dtype=bool)),
        "image_webcam_delta_ms": ("int32", (n_frames,), zeros_i32.copy()),
        # MANO pose
        "generated_mano_left_pose": ("float32", (n_frames, 48), zeros_mano.copy()),
        "generated_mano_right_pose": ("float32", (n_frames, 48), zeros_mano.copy()),
        # Joint angles
        "generated_joint_angles_left": ("float32", (n_frames, 20), zeros_ja),
        "generated_joint_angles_right": ("float32", (n_frames, 20), np.array(joint_angles, dtype=np.float32)),
        # Split
        "frame_split_id": ("int8", (n_frames,), np.zeros(n_frames, dtype=np.int8)),
    }

    # ── Write .dat files ──
    print(f"\nWriting {len(field_data)} field files to {out_dir}/...")

    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "int8": np.int8,
        "bool": bool,
    }

    manifest_fields = {}
    for field_name, (dtype_str, shape, array) in field_data.items():
        filename = f"{field_name}.dat"
        np_dtype = dtype_map[dtype_str]
        arr = np.asarray(array, dtype=np_dtype)
        arr.tofile(out_dir / filename)
        manifest_fields[field_name] = {
            "filename": filename,
            "dtype": dtype_str,
            "shape": list(shape),
        }
        # Print non-trivial fields
        if arr.std() > 0:
            print(f"  {filename}: {arr.shape} {arr.dtype}, "
                  f"mean={arr.mean():.4f}, std={arr.std():.4f}")

    # ── Episode fields ──
    num_episodes = 1
    mano_beta_left = np.zeros((num_episodes, 10), dtype=np.float32)
    mano_beta_right = np.zeros((num_episodes, 10), dtype=np.float32)
    mano_beta_left.tofile(out_dir / "generated_mano_left_beta.dat")
    mano_beta_right.tofile(out_dir / "generated_mano_right_beta.dat")

    # ── Manifest ──
    manifest = {
        "format_version": "egoemg_v2_memmap",
        "total_rows": n_frames,
        "num_episodes": num_episodes,
        "left_hand_strategy": "flip_local_z",
        "mano_label_policy": "generated_only_manus_fit",
        "source_root": str(in_dir.resolve()),
        "fields": manifest_fields,
        "episode_fields": {
            "generated_mano_left_beta": {
                "filename": "generated_mano_left_beta.dat",
                "dtype": "float32",
                "shape": [num_episodes, 10],
            },
            "generated_mano_right_beta": {
                "filename": "generated_mano_right_beta.dat",
                "dtype": "float32",
                "shape": [num_episodes, 10],
            },
        },
        "generated_joint_angles_semantics": [
            "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe",
            "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
            "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
            "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
            "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
        ],
        "frame_split_labels": ["train"],
        "frame_split_policy": "single-session: all frames labeled train",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── metadata.npz ──
    metadata = {
        "episode_id": _str_array([session_name], 32),
        "episode_chunk_id": _str_array(["manus"], 16),
        "episode_subject": _str_array(["manus_subject"], 32),
        "episode_subject_id": np.zeros(num_episodes, dtype=np.int32),
        "episode_source_parquet": _str_array([""], 128),
        "episode_zed_video_path": _str_array([""], 128),
        "episode_webcam_video_path": _str_array([""], 128),
        "episode_start_idx": np.array([0], dtype=np.int64),
        "episode_end_idx": np.array([n_frames - 1], dtype=np.int64),
        "episode_length": np.array([n_frames], dtype=np.int64),
        "episode_beta_idx": np.array([0], dtype=np.int32),
        "subjects_subject": _str_array(["manus_subject"], 32),
        "subjects_subject_id": np.zeros(num_episodes, dtype=np.int32),
        "splits_split": _str_array(["train"], 8),
        "splits_split_id": np.array([0], dtype=np.int32),
    }
    np.savez(out_dir / "metadata.npz", **metadata)

    # ── Summary ──
    summary = {
        "format_version": "egoemg_v2_memmap",
        "source": str(in_dir.resolve()),
        "episodes": num_episodes,
        "total_rows": n_frames,
        "left_hand_strategy": "flip_local_z",
        "mano_label_policy": "generated_only_manus_fit",
        "emg_preprocessing": {
            "fs_hz": FS,
            "scale_factor": args.emg_scale,
            "pipeline": [
                f"divide raw by {args.emg_scale:.0f} to get mV",
                "subtract per-channel mean",
                "narrow notch around 50 Hz",
                "narrow notch around 100 Hz",
                "wide bandpass 20-850 Hz with soft roll-off",
                "no normalization",
            ],
        },
        "generated_joint_angles_semantics": manifest["generated_joint_angles_semantics"],
    }
    with open(out_dir / "metadata_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. EgoEMG v2 memmap written to {out_dir}/")
    print(f"  Total frames: {n_frames}")
    print(f"  Duration: {n_frames / FS:.1f}s ({n_frames / FS / 60:.1f} min)")
    print(f"  Fields: {len(manifest_fields)} frame + 2 episode")
    print(f"\nEMG stats:")
    print(f"  emg_right_raw:      mean={emg_mv.mean():.4f} mV, "
          f"std={emg_mv.std():.4f} mV")
    print(f"  emg_right_filtered: mean={emg_filtered_mv.mean():.4f} mV, "
          f"std={emg_filtered_mv.std():.4f} mV")
    print(f"\nTo use for training:")
    print(f"  manus_memmap_dir={out_dir}")


if __name__ == "__main__":
    main()
