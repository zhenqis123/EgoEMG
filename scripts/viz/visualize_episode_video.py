#!/usr/bin/env python3
"""Visualize mocap reprojection over a full webcam video episode.

Reads world-to-camera transforms and mocap keypoints from the EgoEMG memmap,
projects them onto each webcam frame, and writes an annotated output video.

Usage:
    python scripts/viz/visualize_episode_video.py \
        --memmap-dir data/EgoEMG_v2_memmap \
        --allintra-root data/EgoEMG_allintra \
        --episode-id episode_000006 \
        --output /tmp/ep06_reproj.mp4

    # Subsample: every N-th frame
    python scripts/viz/visualize_episode_video.py \
        --memmap-dir data/EgoEMG_v2_memmap \
        --allintra-root data/EgoEMG_allintra \
        --episode-id episode_000006 \
        --stride 5 \
        --output /tmp/ep06_reproj_s5.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------- intrinsics / projection helpers ----------


def _build_intrinsics_info(frame_bgr: np.ndarray, calib_w: int, calib_h: int):
    """Detect active crop and build intrinsics_info dict (matches dataset code)."""
    video_h, video_w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    col_mean = gray.mean(axis=0)
    active_cols = np.where(col_mean > 2.0)[0]
    if len(active_cols) > 0:
        x0 = int(active_cols[0])
        x1 = int(active_cols[-1]) + 1
    else:
        active_w = int(round(video_h * (calib_w / float(calib_h))))
        x0 = int((video_w - active_w) // 2)
        x1 = x0 + active_w

    return {
        "mode": "gopro_8x7_crop_upsample",
        "crop_xywh_on_video": [x0, 0, x1 - x0, video_h],
        "processed_size": [calib_w, calib_h],
    }


def _project_world_points(points_world, T_W_C, K, dist):
    if len(points_world) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=bool)
    T_C_W = np.linalg.inv(T_W_C)
    R_C_W = T_C_W[:3, :3]
    t_C_W = T_C_W[:3, 3].reshape(3, 1)
    p_cam = (R_C_W @ points_world.T + t_C_W).T
    depth_valid = p_cam[:, 2] > 1e-6
    rvec, _ = cv2.Rodrigues(R_C_W)
    proj, _ = cv2.projectPoints(
        points_world.astype(np.float64), rvec, t_C_W, K, dist,
    )
    return proj.reshape(-1, 2), depth_valid


def _map_processed_to_raw(pts_proc, intrinsics_info):
    if str(intrinsics_info.get("mode")) != "gopro_8x7_crop_upsample":
        return pts_proc.copy()
    crop_xywh = intrinsics_info.get("crop_xywh_on_video")
    processed_size = intrinsics_info.get("processed_size")
    if crop_xywh is None or processed_size is None:
        return pts_proc.copy()
    x0, y0, crop_w, crop_h = [float(v) for v in crop_xywh]
    proc_w, proc_h = [float(v) for v in processed_size]
    out = pts_proc.astype(np.float64).copy()
    out[:, 0] = (out[:, 0] / proc_w) * crop_w + x0
    out[:, 1] = (out[:, 1] / proc_h) * crop_h + y0
    return out


def _decode_transform(t12):
    """12-element float32 -> 4x4 world-to-camera matrix (float64)."""
    t = np.asarray(t12, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = t[:9].reshape(3, 3)
    T[:3, 3] = t[9:12]
    return T


# ---------- drawing ----------

_BODY_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17), (5, 9), (9, 13), (13, 17),       # palm
]

_COLORS = {
    "left":  (0, 0, 255),     # red
    "right": (0, 255, 0),     # green
}


def _draw_hand(frame, pts_raw, pts_valid, color, radius=3):
    for i, (px, py) in enumerate(pts_raw):
        if not pts_valid[i]:
            continue
        cv2.circle(frame, (int(px), int(py)), radius, color, -1)
    for a, b in _BODY_BONES:
        if not (pts_valid[a] and pts_valid[b]):
            continue
        cv2.line(
            frame,
            (int(pts_raw[a, 0]), int(pts_raw[a, 1])),
            (int(pts_raw[b, 0]), int(pts_raw[b, 1])),
            color, 1,
        )


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-dir", required=True)
    parser.add_argument("--allintra-root", required=True)
    parser.add_argument("--episode-id", required=True, help="e.g. episode_000006")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--stride", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=all)")
    parser.add_argument(
        "--calibration-json", default=None,
        help="Path to GX010023 calibration JSON. Auto-detected if not provided.",
    )
    args = parser.parse_args()

    memmap_dir = Path(args.memmap_dir)
    allintra_root = Path(args.allintra_root)

    # Load calibration intrinsics
    if args.calibration_json is not None:
        calib_json = Path(args.calibration_json)
    else:
        # Auto-detect: look in data/EgoEMG/reprojection_assets/
        proj_root = Path(__file__).resolve().parent.parent
        calib_json = proj_root / "data" / "EgoEMG" / "reprojection_assets" / "GX010023_standard_calibration.json"
        if not calib_json.exists():
            raise FileNotFoundError(
                f"Calibration JSON not found at {calib_json}. "
                f"Pass --calibration-json explicitly."
            )

    # Load manifest & metadata
    with open(memmap_dir / "manifest.json") as f:
        manifest = json.load(f)
    fields = manifest["fields"]
    md = np.load(memmap_dir / "metadata.npz", allow_pickle=False)
    ep_ids = [v.decode() if isinstance(v, bytes) else str(v) for v in md["episode_id"]]
    ep_starts = np.asarray(md["episode_start_idx"], dtype=np.int64)
    ep_ends = np.asarray(md["episode_end_idx"], dtype=np.int64)
    ep_paths = [v.decode() for v in md["episode_head_video_path"]]

    ep_idx = ep_ids.index(args.episode_id)
    start = int(ep_starts[ep_idx])
    length = int(ep_ends[ep_idx] - ep_starts[ep_idx])
    raw_rel = Path(ep_paths[ep_idx])
    allintra_path = allintra_root / raw_rel.with_name(f"{raw_rel.stem}_allintra.mp4")

    if not allintra_path.exists():
        raise FileNotFoundError(f"All-intra video not found: {allintra_path}")

    # Open memmaps
    cam_tf_mm = np.memmap(
        memmap_dir / fields["mocap_head_transform"]["filename"],
        dtype=fields["mocap_head_transform"]["dtype"], mode="r",
        shape=tuple(fields["mocap_head_transform"]["shape"]),
    )
    kp_left_mm = np.memmap(
        memmap_dir / fields["mocap_left_keypoints"]["filename"],
        dtype=fields["mocap_left_keypoints"]["dtype"], mode="r",
        shape=tuple(fields["mocap_left_keypoints"]["shape"]),
    )
    kp_right_mm = np.memmap(
        memmap_dir / fields["mocap_right_keypoints"]["filename"],
        dtype=fields["mocap_right_keypoints"]["dtype"], mode="r",
        shape=tuple(fields["mocap_right_keypoints"]["shape"]),
    )
    valid_left_mm = np.memmap(
        memmap_dir / fields["mocap_left_valid"]["filename"],
        dtype=fields["mocap_left_valid"]["dtype"], mode="r",
        shape=tuple(fields["mocap_left_valid"]["shape"]),
    )
    valid_right_mm = np.memmap(
        memmap_dir / fields["mocap_right_valid"]["filename"],
        dtype=fields["mocap_right_valid"]["dtype"], mode="r",
        shape=tuple(fields["mocap_right_valid"]["shape"]),
    )
    frame_idx_mm = np.memmap(
        memmap_dir / fields["image_head_frame_index"]["filename"],
        dtype=fields["image_head_frame_index"]["dtype"], mode="r",
        shape=tuple(fields["image_head_frame_index"]["shape"]),
    )

    with open(calib_json) as f:
        calib = json.load(f)
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["distortion_coefficients"], dtype=np.float64).flatten()
    calib_w = calib["image_width"]
    calib_h = calib["image_height"]

    # Open video reader
    cap = cv2.VideoCapture(str(allintra_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract unique video frame indices and their first memmap frame offsets
    # (memmap runs at 2000Hz, video at ~60fps → ~33 memmap frames per video frame)
    raw_frame_indices = np.asarray(frame_idx_mm[start:start+length], dtype=np.int64)

    # Build a list of (video_frame_idx, memmap_offset) for unique video frames
    seen = {}
    unique_frames = []
    for offset, vfi in enumerate(raw_frame_indices):
        if vfi >= 0 and vfi not in seen:
            seen[vfi] = offset
            unique_frames.append((int(vfi), offset))

    total_video_frames = len(unique_frames)
    print(f"[{args.episode_id}] {total_video_frames} unique video frames in episode "
          f"({length} memmap frames, ~{length / max(total_video_frames, 1):.1f}x oversampling)")

    # Apply stride to video frames (not memmap frames!)
    strided_frames = unique_frames[0::args.stride]
    if args.max_frames > 0:
        strided_frames = strided_frames[:args.max_frames]

    # Write at native FPS — stride=2 with 60fps → 60fps playback, every other frame shown once
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(args.output, fourcc, int(round(fps / args.stride)), (video_w, video_h))

    intrinsics_info = None
    pbar = tqdm(strided_frames, desc=f"Processing {args.episode_id}", unit="vframe")
    for video_frame_idx, memmap_offset in pbar:
        global_i = start + memmap_offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        # Build intrinsics info from first frame
        if intrinsics_info is None:
            intrinsics_info = _build_intrinsics_info(frame, calib_w, calib_h)

        # Decode world-to-camera
        t12 = np.asarray(cam_tf_mm[global_i], dtype=np.float64)
        T_W_C = np.eye(4, dtype=np.float64)
        T_W_C[:3, :3] = t12[:9].reshape(3, 3)
        T_W_C[:3, 3] = t12[9:12]

        # Left hand
        kp_left = np.asarray(kp_left_mm[global_i], dtype=np.float64)
        valid_left = np.asarray(valid_left_mm[global_i], dtype=bool)
        if valid_left.any():
            pts_left_proc, depth_left = _project_world_points(kp_left[valid_left], T_W_C, K, dist)
            pts_left_raw = _map_processed_to_raw(pts_left_proc, intrinsics_info)
            valid_left_mask = depth_left & (pts_left_raw[:, 0] >= 0) & (pts_left_raw[:, 0] < video_w) & (pts_left_raw[:, 1] >= 0) & (pts_left_raw[:, 1] < video_h)
            _draw_hand(frame, pts_left_raw, valid_left_mask, _COLORS["left"])

        # Right hand
        kp_right = np.asarray(kp_right_mm[global_i], dtype=np.float64)
        valid_right = np.asarray(valid_right_mm[global_i], dtype=bool)
        if valid_right.any():
            pts_right_proc, depth_right = _project_world_points(kp_right[valid_right], T_W_C, K, dist)
            pts_right_raw = _map_processed_to_raw(pts_right_proc, intrinsics_info)
            valid_right_mask = depth_right & (pts_right_raw[:, 0] >= 0) & (pts_right_raw[:, 0] < video_w) & (pts_right_raw[:, 1] >= 0) & (pts_right_raw[:, 1] < video_h)
            _draw_hand(frame, pts_right_raw, valid_right_mask, _COLORS["right"])

        # Overlay text
        cv2.putText(frame, f"{args.episode_id}  frame={video_frame_idx}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "L=red  R=green",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
