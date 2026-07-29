"""Verify per-episode projection correctness using raw parquet data.

Reads keypoints and the pre-computed webcam transform from parquet,
projects onto webcam video frames, and saves one annotated image per
episode / hand so the user can visually confirm alignment.

Usage:
    python scripts/prepare/verify_projection_from_parquet.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

# ---------- paths ----------
DATASET_ROOT = Path("data/training_dataset_lerobot_full_NEW")
DATA_DIR = DATASET_ROOT / "data" / "chunk-000"
VIDEO_DIR = DATASET_ROOT / "videos" / "observation.images.webcam" / "chunk-000"
CALIB_PATH = Path("./data/EgoEMG/reprojection_assets/GX010023_standard_calibration.json")
OUTPUT_DIR = Path("./visualizations/verify_parquet_projection")

STRIDE = 30
SAMPLE_OFFSET = 2  # pick the 3rd sample at coarse stride (mid-episode)


def _load_calibration():
    with CALIB_PATH.open("r") as f:
        calib = json.load(f)
    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    dist = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    cw = int(calib["image_width"])
    ch = int(calib["image_height"])
    return K, dist, cw, ch


def _detect_active_crop(frame_bgr):
    """Detect the active (non-black) region of a GoPro 1280x720 frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    col_mean = gray.mean(axis=0)
    active = np.where(col_mean > 2.0)[0]
    if len(active) > 0:
        return int(active[0]), int(active[-1]) + 1
    return 0, frame_bgr.shape[1]


def _build_intrinsics(K, dist, calib_w, calib_h, video_w, video_h, x0, x1):
    """Scale calibration intrinsics from calib resolution to cropped video region."""
    active_w = x1 - x0
    sx = calib_w / float(active_w)
    sy = calib_h / float(video_h)
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K2[0, 2] - x0) * sx
    K2[1, 2] = K2[1, 2] * sy
    crop_info = {"x0": x0, "x1": x1, "active_w": active_w, "sx": sx, "sy": sy}
    return K2, dist.copy(), crop_info


def _project(points_world, T_W_C, K, dist):
    """Project world points to image pixel coordinates."""
    T_C_W = np.linalg.inv(T_W_C)
    R = T_C_W[:3, :3]
    t = T_C_W[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(points_world.astype(np.float64), rvec, t, K, dist)
    return proj.reshape(-1, 2)


def _map_to_raw(pts_crop, crop_info, video_w, video_h):
    """Map from cropped-and-upsampled coords back to raw 1280x720 coords."""
    sx = crop_info["sx"]
    sy = crop_info["sy"]
    x0 = crop_info["x0"]
    out = pts_crop.copy().astype(np.float64)
    out[:, 0] = out[:, 0] / sx + x0
    out[:, 1] = out[:, 1] / sy
    return out


def _read_parquet_columns(pq_path, columns):
    table = pq.read_table(pq_path, columns=columns)
    arrays = {}
    for col_name in columns:
        col = table.column(col_name).combine_chunks()
        if col.type.equals(pyarrow_fixed_size_list_12):
            flat = col.flatten().to_numpy(zero_copy_only=False)
            arrays[col_name] = flat.reshape(len(col), 12)
        else:
            arrays[col_name] = _column_to_numpy(col)
    return arrays


import pyarrow

fixed_size_list_12 = pyarrow.list_(pyarrow.float32(), 12)

def _column_to_numpy(col):
    combined = col.combine_chunks()
    flat = combined.flatten().to_numpy(zero_copy_only=False)
    first = combined[0].as_py()
    if isinstance(first, list):
        if isinstance(first[0], list):
            inner2 = len(first[0])
            inner1 = len(first)
            return flat.reshape(-1, inner1, inner2)
        return flat.reshape(-1, len(first))
    return flat


def _read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    K, dist, calib_w, calib_h = _load_calibration()

    pq_files = sorted(DATA_DIR.glob("episode_*.parquet"))
    print(f"Found {len(pq_files)} episode parquets")

    for pq_path in pq_files:
        ep_stem = pq_path.stem
        out_dir = OUTPUT_DIR / ep_stem
        if (out_dir / f"{ep_stem}_left.png").exists() and (out_dir / f"{ep_stem}_right.png").exists():
            print(f"[{ep_stem}] already done, skip")
            continue

        table = pq.read_table(str(pq_path))
        n_rows = table.num_rows
        if n_rows < 100:
            print(f"[{ep_stem}] too short ({n_rows}), skip")
            continue

        # Pick a sample index near the middle of the episode
        indices = list(range(0, n_rows, STRIDE))
        if len(indices) <= SAMPLE_OFFSET:
            print(f"[{ep_stem}] not enough samples, skip")
            continue
        sample_idx = indices[SAMPLE_OFFSET]

        # Read required columns at the sample index
        col_names = [
            "observation.mocap.hand.left.keypoints",
            "observation.mocap.hand.right.keypoints",
            "observation.mocap.hand.left.valid",
            "observation.mocap.hand.right.valid",
            "observation.images.webcam.frame_index",
            "observation.mocap.webcam.transform",
        ]

        df = table.to_pandas()
        left_kp = np.stack(df["observation.mocap.hand.left.keypoints"].values)[sample_idx]
        right_kp = np.stack(df["observation.mocap.hand.right.keypoints"].values)[sample_idx]
        left_valid = np.stack(df["observation.mocap.hand.left.valid"].values)[sample_idx]
        right_valid = np.stack(df["observation.mocap.hand.right.valid"].values)[sample_idx]
        wc_frame_idx = int(np.asarray(df["observation.images.webcam.frame_index"].values)[sample_idx])
        transform_12 = np.asarray(df["observation.mocap.webcam.transform"].values)[sample_idx].astype(np.float64)

        # Reconstruct T_W_C from the 12-element flat representation
        T_W_C = np.eye(4, dtype=np.float64)
        T_W_C[:3, :3] = transform_12[:9].reshape(3, 3)
        T_W_C[:3, 3] = transform_12[9:12]

        # Read video frame
        video_path = VIDEO_DIR / f"{ep_stem}.mp4"
        if not video_path.exists():
            print(f"[{ep_stem}] video not found: {video_path}")
            continue

        frame_bgr = _read_video_frame(video_path, wc_frame_idx)
        vh, vw = frame_bgr.shape[:2]

        # Build intrinsics for GoPro crop
        x0, x1 = _detect_active_crop(frame_bgr)
        K_use, dist_use, crop_info = _build_intrinsics(K, dist, calib_w, calib_h, vw, vh, x0, x1)

        # Project keypoints
        left_proj = _project(left_kp, T_W_C, K_use, dist_use)
        right_proj = _project(right_kp, T_W_C, K_use, dist_use)

        # Map from cropped-upscaled coords to raw video coords
        left_raw = _map_to_raw(left_proj, crop_info, vw, vh)
        right_raw = _map_to_raw(right_proj, crop_info, vw, vh)

        # Draw on the raw frame
        canvas = frame_bgr.copy()
        colors = {"left": (0, 220, 255), "right": (255, 180, 0)}  # BGR: yellow, cyan

        for side, pts, valid in [("left", left_raw, left_valid), ("right", right_raw, right_valid)]:
            for i, (pt, v) in enumerate(zip(pts, valid)):
                if not v:
                    continue
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < vw and 0 <= y < vh:
                    cv2.circle(canvas, (x, y), 3, colors[side], -1)
                    cv2.putText(canvas, f"{side[0].upper()}{i}", (x + 4, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, colors[side], 1, cv2.LINE_AA)

        info_text = f"{ep_stem} frame={wc_frame_idx} idx={sample_idx}"
        cv2.putText(canvas, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        out_dir.mkdir(parents=True, exist_ok=True)
        # Save combined image
        cv2.imwrite(str(out_dir / f"{ep_stem}_both.png"), canvas)
        print(f"[{ep_stem}] saved  idx={sample_idx} wc_frame={wc_frame_idx}")

    print("Done.")


if __name__ == "__main__":
    main()
