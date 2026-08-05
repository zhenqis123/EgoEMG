"""Verify ShowEE session-frame alignment: video frame vs projected markers.

For a ShowEE session episode (session-level 71-episode layout), pick
mid-action rows, decode the session all-intra video at the corrected
``image_head_frame_index`` value, project the mocap wrist markers onto
the frame with the ShowEE camera calibration, and write an overlay PNG
per sample.  The markers should land on the hands visible in the frame.

Usage::

    python scripts/viz/verify_showee_session_alignment.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_unified_memmap \
        --allintra-root /mnt/nvme/xiziheng/EgoEMG_allintra \
        --calibration-path /mnt/nvme/xiziheng/showee_calibration.json \
        --output-dir /tmp/showee_align_check \
        --session-index 41 \
        --samples-per-action 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader, cpu

from egoemg.datasets.egoemg_vision_dataset import (
    _build_intrinsics_and_frame_mapper,
    _map_processed_points_to_raw,
    _project_world_points,
)
from egoemg.video_io import resolve_allintra_video_path


def _clean(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for v in values:
        s = v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
        s = s.strip("b'").strip('"')
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--allintra-root", type=Path, required=True)
    ap.add_argument("--calibration-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--session-index", type=int, required=True)
    ap.add_argument("--samples-per-action", type=int, default=3)
    args = ap.parse_args()

    md = np.load(args.memmap_dir / "metadata.npz", allow_pickle=False)
    with (args.memmap_dir / "manifest.json").open() as f:
        manifest = json.load(f)
    n_rows = int(manifest["fields"]["image_head_frame_index"]["shape"][0])
    fi = np.memmap(
        args.memmap_dir / manifest["fields"]["image_head_frame_index"]["filename"],
        dtype=manifest["fields"]["image_head_frame_index"]["dtype"],
        mode="r", shape=(n_rows,),
    )

    def mm(name: str) -> np.memmap:
        info = manifest["fields"][name]
        return np.memmap(args.memmap_dir / info["filename"], mode="r",
                         dtype=info["dtype"], shape=tuple(info["shape"]))

    ep_idx = args.session_index
    s, e = int(md["episode_start_idx"][ep_idx]), int(md["episode_end_idx"][ep_idx])
    ep_id = _clean(np.asarray([md["episode_id"][ep_idx]]))[0]
    raw_video = _clean(np.asarray([md["episode_head_video_path"][ep_idx]]))[0]
    video_path = resolve_allintra_video_path(
        raw_video_path=raw_video, data_root=args.memmap_dir.parent,
        allintra_root=args.allintra_root,
    )
    print(f"episode {ep_idx} ({ep_id}): rows [{s}, {e}), video {video_path}")

    with args.calibration_path.open() as f:
        calib = json.load(f)
    K_calib = np.asarray(calib["camera_matrix"], dtype=np.float64)
    dist_calib = np.asarray(
        calib["distortion_coefficients"], dtype=np.float64
    ).reshape(-1, 1)
    calib_w, calib_h = int(calib["image_width"]), int(calib["image_height"])

    vr = VideoReader(str(video_path), ctx=cpu(0))
    n_video_frames = len(vr)
    frame0 = vr[0].asnumpy()
    K, dist, info = _build_intrinsics_and_frame_mapper(
        K_calib, dist_calib, calib_w, calib_h, frame0.shape[1], frame0.shape[0], frame0,
    )
    video_h, video_w = frame0.shape[:2]
    print(f"video: {n_video_frames} frames, {video_w}x{video_h}")

    transforms = mm("mocap_head_transform")
    kp_l = mm("mocap_left_keypoints")
    kp_r = mm("mocap_right_keypoints")
    valid_l = mm("mocap_left_valid")
    valid_r = mm("mocap_right_valid")
    lv = mm("generated_label_valid")
    stale = mm("image_head_stale")

    # Sample rows evenly across the episode, skipping stale frames.
    rows = np.linspace(s, e - 1, args.samples_per_action * 40).astype(np.int64)
    rows = [r for r in rows if not bool(stale[r]) and bool(lv[r].all())][
        : args.samples_per_action * 8
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        frame_idx = int(fi[r])
        if frame_idx < 0 or frame_idx >= n_video_frames:
            print(f"  row {r}: frame index {frame_idx} out of video range, skip")
            continue
        frame = np.ascontiguousarray(vr[frame_idx].asnumpy()[:, :, ::-1])  # BGR
        t12 = np.asarray(transforms[r]).reshape(-1)
        T_W_C = np.eye(4, dtype=np.float64)
        T_W_C[:3, :3] = t12[:9].reshape(3, 3)
        T_W_C[:3, 3] = t12[9:12]

        for hand_name, kp_mm, valid_mm, color in (
            ("left", kp_l, valid_l, (0, 255, 0)),
            ("right", kp_r, valid_r, (0, 0, 255)),
        ):
            marker_world = np.asarray(kp_mm[r], dtype=np.float64)
            marker_valid = np.asarray(valid_mm[r], dtype=bool)
            proc, depth_valid = _project_world_points(marker_world, T_W_C, K, dist)
            raw = _map_processed_points_to_raw(proc, info)
            in_img = (
                (raw[:, 0] >= 0) & (raw[:, 0] < video_w)
                & (raw[:, 1] >= 0) & (raw[:, 1] < video_h)
            )
            good = marker_valid & depth_valid & in_img
            for x, y, ok in zip(raw[:, 0], raw[:, 1], good):
                if ok:
                    cv2.circle(frame, (int(x), int(y)), 6, color, -1)
        cv2.putText(frame, f"row={r} video_frame={frame_idx}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        out = args.output_dir / f"row_{r}_vf_{frame_idx}.png"
        cv2.imwrite(str(out), frame)
        print(f"  row {r} -> {out}")

    print(f"\nWrote {len(rows)} overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
