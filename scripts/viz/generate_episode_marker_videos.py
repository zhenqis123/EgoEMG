"""Generate per-episode marker projection videos (10fps, both hands).

For each episode: project left & right mocap markers onto the original webcam
frames, draw skeletons, and export as an mp4.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
MEMMAP_DIR = "./data/EgoEMG_memmap"
VIDEO_ROOT = "data/training_dataset_lerobot_full_NEW"
DATA_ROOT = "data/training_dataset_lerobot_full_NEW"

if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

SKELETON_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

# ── Caches ────────────────────────────────────────────────────────────────────
_MEM_CACHE: dict = {}
_MANIFEST = None
_MD = None
_CALIB = None
_INTRINSICS_CACHE = {}
_VIDEO_CACHE: dict = {}


def _manifest():
    global _MANIFEST
    if _MANIFEST is None:
        with open(Path(MEMMAP_DIR) / "manifest.json") as f:
            _MANIFEST = json.load(f)
    return _MANIFEST


def _metadata():
    global _MD
    if _MD is None:
        _MD = dict(np.load(Path(MEMMAP_DIR) / "metadata.npz", allow_pickle=False))
    return _MD


def _load_mm(name: str) -> np.memmap:
    if name not in _MEM_CACHE:
        mf = _manifest()
        if name in mf.get("fields", {}):
            info = mf["fields"][name]
        else:
            info = mf["episode_fields"][name]
        _MEM_CACHE[name] = np.memmap(
            Path(MEMMAP_DIR) / info["filename"],
            dtype=np.dtype(info["dtype"]),
            mode="r",
            shape=tuple(info["shape"]),
        )
    return _MEM_CACHE[name]


def _get_calib():
    global _CALIB
    if _CALIB is None:
        cp = Path(DATA_ROOT) / "reprojection_assets" / "GX010023_standard_calibration.json"
        with open(cp, "r", encoding="utf-8") as f:
            _CALIB = json.load(f)
    return _CALIB


def _get_intrinsics(video_w: int, video_h: int):
    key = (video_w, video_h)
    if key not in _INTRINSICS_CACHE:
        from reproject_hand_keypoints import build_intrinsics_and_frame_mapper
        calib = _get_calib()
        K_raw = np.asarray(calib["camera_matrix"], dtype=np.float64)
        dist_raw = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
        calib_w = int(calib["image_width"])
        calib_h = int(calib["image_height"])
        K, dist, info, _ = build_intrinsics_and_frame_mapper(
            K_raw, dist_raw, calib_w, calib_h, video_w, video_h,
            mode="gopro_8x7_crop_upsample",
        )
        _INTRINSICS_CACHE[key] = (K, dist, info)
    return _INTRINSICS_CACHE[key]


def get_camera_transform(abs_idx: int) -> np.ndarray:
    mm = _load_mm("mocap_head_transform")
    t12 = np.asarray(mm[abs_idx], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = t12[:9].reshape(3, 3)
    T[:3, 3] = t12[9:12]
    return T


def project_markers(pts_world: np.ndarray, T_W_C: np.ndarray,
                    K: np.ndarray, dist: np.ndarray, info: dict,
                    img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    from reproject_hand_keypoints import _project_world_points, _map_processed_points_to_raw
    pts_proc, depth_valid = _project_world_points(pts_world, T_W_C, K, dist)
    pts_raw = _map_processed_points_to_raw(pts_proc, info)
    in_img = (pts_raw[:, 0] >= 0) & (pts_raw[:, 0] < img_w) & \
             (pts_raw[:, 1] >= 0) & (pts_raw[:, 1] < img_h)
    return pts_raw, depth_valid & in_img & np.isfinite(pts_world).all(axis=1)


def draw_skeleton_2d(img_bgr: np.ndarray, pts: np.ndarray, valid: np.ndarray,
                      color: tuple, label: str) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    # Bones
    for i0, i1 in SKELETON_EDGES:
        if i0 >= len(pts) or i1 >= len(pts):
            continue
        if valid[i0] and valid[i1]:
            p0 = tuple(np.round(pts[i0]).astype(np.int32))
            p1 = tuple(np.round(pts[i1]).astype(np.int32))
            cv2.line(img_bgr, p0, p1, color, 2, lineType=cv2.LINE_AA)
    # Joints
    for i, (p, v) in enumerate(zip(pts, valid)):
        if not v:
            continue
        center = tuple(np.round(p).astype(np.int32))
        cv2.circle(img_bgr, center, 3, color, -1, lineType=cv2.LINE_AA)
    # Label
    if valid.any():
        cy, cx = pts[valid].mean(axis=0).astype(np.int32)
        cv2.putText(img_bgr, label, (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img_bgr


def get_episode_frame_range(ep_idx: int, ep_start: int, ep_end: int,
                              stride: int = 600) -> list[int]:
    """Return global memmap indices for this episode, subsampled at stride.
    Default stride=600 → 2000Hz/600≈3.3 samples/s, at 10fps playback ≈3x speed."""
    ep_mm = _load_mm("episode_index")
    # frame indices belonging to this episode
    mask = np.asarray(ep_mm[ep_start:ep_end]).ravel() == ep_idx
    global_offsets = np.where(mask)[0]
    # subsample (every 600th sample → ~3.3fps source, 10fps output ≈ 3x real-time)
    sampled = global_offsets[::stride]
    return [ep_start + int(o) for o in sampled]


def detect_video_frozen_frames(video_path: str, threshold: float = 2.0) -> set:
    """Return set of video frame indices that are near-duplicates of their predecessor.

    Pre-scans the video to find frames that are nearly identical to the previous
    frame (mean absolute pixel difference < threshold). These are likely from
    webcam recording freezes where the visual content stalls but EMG/marker
    sampling continues. Only the *subsequent* frames in each frozen run are
    returned — the first frame of each run is considered valid.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frozen = set()
    if total < 2:
        cap.release()
        return frozen

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return frozen
    prev_f32 = prev.astype(np.float32)

    frame_idx = 1
    while True:
        ret, cur = cap.read()
        if not ret:
            break
        diff = np.abs(prev_f32 - cur.astype(np.float32)).mean()
        if diff < threshold:
            frozen.add(frame_idx)
        prev_f32 = cur.astype(np.float32)
        frame_idx += 1

    cap.release()
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./visualizations/episode_videos")
    parser.add_argument("--episodes", type=str, nargs="*", default=None,
                        help="Episode IDs to process (default: all)")
    parser.add_argument("--stride", type=int, default=600,
                        help="Subsampling stride (2000Hz/600≈3.3 samples/s, at 10fps ≈3x real-time)")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--limit-frames", type=int, default=0,
                        help="Max frames per episode (0=all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    md = _metadata()
    episode_ids = [x.decode("utf-8").rstrip("\x00") if isinstance(x, (bytes, np.bytes_)) else str(x)
                   for x in md["episode_id"]]
    ep_start_idx = np.asarray(md["episode_start_idx"], dtype=np.int64)
    total_frames = len(_load_mm("episode_index"))

    # Determine which episodes to process
    if args.episodes:
        target_eps = [(i, eid) for i, eid in enumerate(episode_ids) if eid in args.episodes]
    else:
        target_eps = list(enumerate(episode_ids))

    print(f"Processing {len(target_eps)} episodes, stride={args.stride} (output {args.fps}fps)")

    cam_tracked_mm = _load_mm("mocap_head_tracked")

    for ep_idx, ep_id in target_eps:
        ep_start = int(ep_start_idx[ep_idx])
        ep_end = int(ep_start_idx[ep_idx + 1]) if ep_idx + 1 < len(ep_start_idx) else total_frames

        frame_indices = get_episode_frame_range(ep_idx, ep_start, ep_end, args.stride)
        if args.limit_frames > 0:
            frame_indices = frame_indices[:args.limit_frames]
        if len(frame_indices) == 0:
            print(f"  {ep_id}: no frames, skip")
            continue

        # Open video for this episode
        video_rel = [x.decode("utf-8").rstrip("\x00") if isinstance(x, (bytes, np.bytes_)) else str(x)
                     for x in md["episode_head_video_path"]][ep_idx]
        video_path = Path(VIDEO_ROOT) / video_rel
        cap = cv2.VideoCapture(str(video_path))
        webcam_frame_idx_mm = _load_mm("image_head_frame_index")

        # Get frame dimensions from first frame
        ret, test_frame = cap.read()
        if not ret:
            cap.release()
            print(f"  {ep_id}: cannot read video, skip")
            continue
        out_h, out_w = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out_path = os.path.join(args.output_dir, f"{ep_id}_markers.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264, VSCode-playable
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (out_w, out_h))
        if not writer.isOpened():
            # Fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (out_w, out_h))

        K, dist, info = _get_intrinsics(out_w, out_h)

        # Pre-load marker memmaps
        kp_left_mm = _load_mm("mocap_left_keypoints")
        kp_right_mm = _load_mm("mocap_right_keypoints")

        # Build render list: (abs_idx, video_frame_id) sorted by video_frame_id
        # for sequential reading. Filter untracked frames.
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        render_list = []
        for abs_idx in frame_indices:
            if not bool(cam_tracked_mm[abs_idx]):
                continue
            vfid = int(webcam_frame_idx_mm[abs_idx])
            vfid = max(0, min(vfid, total_video_frames - 1))
            render_list.append((abs_idx, vfid))
        # frame_indices are increasing → vfids are non-decreasing; sort to be safe
        render_list.sort(key=lambda x: x[1])

        written = 0
        cur_vfid = -1
        last_written_frame = None  # for duplicate detection
        dup_skip_thresh = 0.5  # mean pixel diff below this = frozen/duplicate
        pbar = tqdm(render_list, desc=f"  {ep_id}", unit="f")

        for abs_idx, vfid in pbar:
            # Read forward to target video frame (skip frames in between)
            while cur_vfid < vfid:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                cur_vfid += 1
            if cur_vfid != vfid:  # couldn't reach target frame
                continue

            # Skip if this video frame is near-identical to the last written one
            # (video recording freeze → markers keep moving, but we skip duplicate frames)
            if last_written_frame is not None:
                frame_diff = np.abs(last_written_frame.astype(np.float32) -
                                    frame_bgr.astype(np.float32)).mean()
                if frame_diff < dup_skip_thresh:
                    pbar.set_postfix(skipped_dup=written)
                    continue

            # Save original for next duplicate check (before drawing overlay)
            original_bgr = frame_bgr.copy()

            T_W_C = get_camera_transform(abs_idx)

            # Left hand markers
            kp_left = np.asarray(kp_left_mm[abs_idx], dtype=np.float64)
            if np.isfinite(kp_left).all() and np.abs(kp_left).sum() > 0:
                pts_left, valid_left = project_markers(kp_left, T_W_C, K, dist, info, out_w, out_h)
                if valid_left.sum() > 0:
                    frame_bgr = draw_skeleton_2d(frame_bgr, pts_left, valid_left,
                                                  color=(0, 255, 255), label="L")

            # Right hand markers
            kp_right = np.asarray(kp_right_mm[abs_idx], dtype=np.float64)
            if np.isfinite(kp_right).all() and np.abs(kp_right).sum() > 0:
                pts_right, valid_right = project_markers(kp_right, T_W_C, K, dist, info, out_w, out_h)
                if valid_right.sum() > 0:
                    frame_bgr = draw_skeleton_2d(frame_bgr, pts_right, valid_right,
                                                  color=(255, 0, 255), label="R")

            # Overlay frame info
            cv2.putText(frame_bgr, f"{ep_id}  frame={vfid}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"{ep_id}  frame={vfid}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (20, 20, 20), 1, cv2.LINE_AA)

            writer.write(frame_bgr)
            last_written_frame = original_bgr
            written += 1

        pbar.close()
        cap.release()
        writer.release()

        if written > 0:
            print(f"    -> {out_path} ({written} frames, {written/args.fps:.1f}s)")
        else:
            print(f"    -> no frames written")
            os.remove(out_path)


if __name__ == "__main__":
    main()
