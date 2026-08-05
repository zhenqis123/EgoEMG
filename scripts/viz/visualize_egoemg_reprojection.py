"""Visualize one reprojection sample per EgoEMG episode.

Reads mocap keypoints from the memmap, projects them onto the corresponding
webcam frame using the stored world-to-camera transform, and saves one PNG
per episode to ``--output-dir``.

Usage:
    python scripts/viz/visualize_egoemg_reprojection.py \
        --memmap-dir data/EgoEMG_memmap \
        --allintra-root data/EgoEMG_allintra \
        --calibration-json data/EgoEMG/reprojection_assets/GX010023_standard_calibration.json \
        --output-dir /tmp/egoemg_reprojection
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ---------- helpers ----------


def decode_bytes(arr: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _detect_active_crop(frame_bgr: np.ndarray) -> tuple[int, int]:
    """Return (x0, x1) of the non-black column region."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    col_mean = gray.mean(axis=0)
    active = np.where(col_mean > 2.0)[0]
    return int(active[0]), int(active[-1]) + 1


def _project_world_points(
    pts_world: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T_C_W = np.linalg.inv(T_W_C)
    R_C_W = T_C_W[:3, :3]
    t_C_W = T_C_W[:3, 3].reshape(3, 1)
    p_cam = (R_C_W @ pts_world.T + t_C_W).T
    depth_valid = p_cam[:, 2] > 1e-6
    rvec, _ = cv2.Rodrigues(R_C_W)
    proj, _ = cv2.projectPoints(
        pts_world.astype(np.float64), rvec, t_C_W, K, dist,
    )
    return proj.reshape(-1, 2), depth_valid


def _map_to_raw(
    pts: np.ndarray,
    crop_x0: int,
    crop_x1: int,
    video_h: int,
    calib_w: int,
    calib_h: int,
) -> np.ndarray:
    crop_w = crop_x1 - crop_x0
    out = pts.astype(np.float64).copy()
    out[:, 0] = out[:, 0] / calib_w * crop_w + crop_x0
    out[:, 1] = out[:, 1] / calib_h * video_h
    return out


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One reprojection image per EgoEMG episode",
    )
    parser.add_argument(
        "--memmap-dir", required=True, help="EgoEMG memmap directory",
    )
    parser.add_argument(
        "--allintra-root", required=True, help="EgoEMG all-intra video root",
    )
    parser.add_argument(
        "--calibration-json", required=True,
        help="Path to GX010023_standard_calibration.json",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for PNGs",
    )
    parser.add_argument(
        "--frame-fraction", type=float, default=0.3,
        help="Fraction into the episode for the sample frame (default 0.3)",
    )
    args = parser.parse_args()

    memmap_dir = Path(args.memmap_dir)
    allintra_root = Path(args.allintra_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration
    with open(args.calibration_json) as f:
        calib = json.load(f)
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["distortion_coefficients"], dtype=np.float64).flatten()
    calib_w = calib["image_width"]
    calib_h = calib["image_height"]

    # Load memmap metadata
    with open(memmap_dir / "manifest.json") as f:
        manifest = json.load(f)
    fields = manifest["fields"]
    num_episodes = manifest["num_episodes"]

    md = np.load(memmap_dir / "metadata.npz", allow_pickle=True)
    ep_start = md["episode_start_idx"]
    ep_length = md["episode_length"]
    ep_webcam_path = decode_bytes(md["episode_head_video_path"])

    # Open memmap arrays
    cam_tf_mm = np.memmap(
        memmap_dir / fields["mocap_head_transform"]["filename"],
        dtype=fields["mocap_head_transform"]["dtype"],
        mode="r",
        shape=tuple(fields["mocap_head_transform"]["shape"]),
    )
    kp_left_mm = np.memmap(
        memmap_dir / fields["mocap_left_keypoints"]["filename"],
        dtype=fields["mocap_left_keypoints"]["dtype"],
        mode="r",
        shape=tuple(fields["mocap_left_keypoints"]["shape"]),
    )
    kp_right_mm = np.memmap(
        memmap_dir / fields["mocap_right_keypoints"]["filename"],
        dtype=fields["mocap_right_keypoints"]["dtype"],
        mode="r",
        shape=tuple(fields["mocap_right_keypoints"]["shape"]),
    )
    img_fi_mm = np.memmap(
        memmap_dir / fields["image_head_frame_index"]["filename"],
        dtype=fields["image_head_frame_index"]["dtype"],
        mode="r",
        shape=tuple(fields["image_head_frame_index"]["shape"]),
    )

    def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        return frame

    for ep in range(num_episodes):
        start = int(ep_start[ep])
        length = int(ep_length[ep])
        if length == 0:
            continue

        # Pick a frame: use the frame at given fraction of the episode
        local_idx = min(int(length * args.frame_fraction), length - 1)
        global_idx = start + local_idx

        # Resolve all-intra video path
        raw_rel = Path(ep_webcam_path[ep])
        allintra_path = allintra_root / raw_rel.with_name(
            f"{raw_rel.stem}_allintra.mp4",
        )
        if not allintra_path.exists():
            print(f"[ep {ep}] SKIP: video not found: {allintra_path}")
            continue

        # Read video frame via OpenCV
        video_frame_idx = int(img_fi_mm[global_idx])
        frame_bgr = _read_frame(allintra_path, video_frame_idx)
        video_h, video_w = frame_bgr.shape[:2]

        # Detect active crop
        crop_x0, crop_x1 = _detect_active_crop(frame_bgr)

        # Decode world-to-camera transform
        t12 = np.asarray(cam_tf_mm[global_idx], dtype=np.float64)
        T_W_C = np.eye(4, dtype=np.float64)
        T_W_C[:3, :3] = t12[:9].reshape(3, 3)
        T_W_C[:3, 3] = t12[9:12]

        # Get keypoints (already in world coords)
        kp_left = np.asarray(kp_left_mm[global_idx], dtype=np.float64)
        kp_right = np.asarray(kp_right_mm[global_idx], dtype=np.float64)

        # Project and draw
        for kp, color, label in [
            (kp_right, (0, 255, 0), "right"),
            (kp_left, (0, 0, 255), "left"),
        ]:
            finite = np.isfinite(kp).all(axis=1)
            if not finite.any():
                continue
            proj, depth_valid = _project_world_points(
                kp[finite], T_W_C, K, dist,
            )
            proj_raw = _map_to_raw(
                proj, crop_x0, crop_x1, video_h, calib_w, calib_h,
            )
            valid = depth_valid & (proj_raw[:, 0] >= 0) & (proj_raw[:, 0] < video_w) \
                & (proj_raw[:, 1] >= 0) & (proj_raw[:, 1] < video_h)
            for i, (px, py) in enumerate(proj_raw):
                if not valid[i]:
                    continue
                cv2.circle(frame_bgr, (int(px), int(py)), 4, color, -1)

        subject = md["episode_subject"][ep]
        if isinstance(subject, bytes):
            subject = subject.decode()
        out_path = out_dir / f"episode_{ep:03d}_{subject}.png"
        cv2.imwrite(str(out_path), frame_bgr)
        print(f"[ep {ep}] saved {out_path}  (frame {video_frame_idx})")

    print(f"Done. {num_episodes} episodes -> {out_dir}")


if __name__ == "__main__":
    main()
