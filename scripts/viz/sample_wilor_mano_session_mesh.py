#!/usr/bin/env python3
"""Sample WiLoR MANO meshes from a capture session and project them to video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
import trimesh

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
except ImportError:
    WiLorHandPose3dEstimationPipeline = None


def _load_wilor_arrays(wilor_dir: Path) -> dict[str, np.ndarray]:
    return {
        "valid": np.load(wilor_dir / "valid.npy"),
        "frame_indices": np.load(wilor_dir / "frame_indices.npy"),
        "pose": np.concatenate(
            [
                np.load(wilor_dir / "mano_global_orient.npy"),
                np.load(wilor_dir / "mano_hand_pose.npy"),
            ],
            axis=1,
        ).astype(np.float32),
        "betas": np.load(wilor_dir / "mano_betas.npy").astype(np.float32),
        "transl": np.load(wilor_dir / "mano_transl.npy").astype(np.float32),
    }


def _project_vertices(
    vertices_cam: np.ndarray,
    focal: float,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = vertices_cam[:, 2]
    depth_valid = z > 1e-6
    points = np.empty((vertices_cam.shape[0], 2), dtype=np.float32)
    points[:, 0] = vertices_cam[:, 0] / z * focal + image_w / 2.0
    points[:, 1] = vertices_cam[:, 1] / z * focal + image_h / 2.0
    in_image = (
        depth_valid
        & (points[:, 0] >= 0)
        & (points[:, 0] < image_w)
        & (points[:, 1] >= 0)
        & (points[:, 1] < image_h)
    )
    return points, in_image


def _draw_wireframe(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    depth_valid: np.ndarray,
    faces: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    out = frame_bgr.copy()
    image_h, image_w = out.shape[:2]
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (depth_valid[i0] and depth_valid[i1] and depth_valid[i2]):
            continue
        pts = np.round(points[[i0, i1, i2]]).astype(np.int32)
        if (
            (pts[:, 0] < -1000).any()
            or (pts[:, 0] > image_w + 1000).any()
            or (pts[:, 1] < -1000).any()
            or (pts[:, 1] > image_h + 1000).any()
        ):
            continue
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), color_bgr, 1, cv2.LINE_AA)
        cv2.line(out, tuple(pts[1]), tuple(pts[2]), color_bgr, 1, cv2.LINE_AA)
        cv2.line(out, tuple(pts[2]), tuple(pts[0]), color_bgr, 1, cv2.LINE_AA)
    return out


def _draw_yolo_detections(
    frame_bgr: np.ndarray,
    detector,
    target_hand: str,
    conf: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_h, image_w = frame_rgb.shape[:2]
    yolo_h = 256
    yolo_w = max(int(image_w * yolo_h / image_h), 1)
    scale_x = image_w / float(yolo_w)
    scale_y = image_h / float(yolo_h)
    frame_yolo = cv2.resize(frame_rgb, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)

    result = detector(frame_yolo, conf=conf, verbose=False)
    if isinstance(result, list):
        result = result[0]

    out = frame_bgr.copy()
    detections: list[dict[str, object]] = []
    target_is_right = target_hand == "right"
    for det in result:
        is_right = bool(int(det.boxes.cls.cpu().item()))
        box = det.boxes.data.cpu().squeeze().numpy()
        x1, y1, x2, y2 = (
            float(box[0] * scale_x),
            float(box[1] * scale_y),
            float(box[2] * scale_x),
            float(box[3] * scale_y),
        )
        score = float(box[4]) if box.shape[0] > 4 else float("nan")
        color = (0, 255, 0) if is_right else (255, 0, 0)
        label = "YOLO R" if is_right else "YOLO L"
        if is_right == target_is_right:
            label += " TARGET"
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, pt1, pt2, color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"{label} {score:.2f}",
            (pt1[0], max(20, pt1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        detections.append(
            {
                "hand": "right" if is_right else "left",
                "score": score,
                "bbox_xyxy": [x1, y1, x2, y2],
                "is_target": bool(is_right == target_is_right),
            }
        )
    return out, detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--mano-model-path",
        type=Path,
        default=Path("/home/xiziheng/develop/WiLoR/mano_data/models"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=140912)
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="Comma-separated video frame indices to visualize instead of random samples.",
    )
    parser.add_argument(
        "--sequential-video-read",
        action="store_true",
        help="Read video frames sequentially instead of using cap.set(frame_idx).",
    )
    parser.add_argument("--focal-length", type=float, default=5000.0)
    parser.add_argument("--image-size", type=float, default=256.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--draw-yolo", action="store_true")
    parser.add_argument("--target-hand", default="right", choices=["right", "left"])
    parser.add_argument("--yolo-conf", type=float, default=0.3)
    parser.add_argument(
        "--detector-path",
        type=Path,
        default=Path("data/pretrained_models/detector.pt"),
    )
    parser.add_argument("--wilor-pretrained-dir", default=None)
    args = parser.parse_args()

    session = args.session
    wilor_dir = session / "wilor_mano"
    output_dir = args.output or wilor_dir / "random_mesh_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    zed_dirs = sorted(p for p in session.iterdir() if p.is_dir() and p.name.startswith("ZED_"))
    if not zed_dirs:
        raise FileNotFoundError(f"No ZED_* directory under {session}")
    video_path = zed_dirs[0] / "rgb.mkv"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    arrays = _load_wilor_arrays(wilor_dir)
    candidates = np.where(
        arrays["valid"]
        & np.isfinite(arrays["pose"]).all(axis=1)
        & np.isfinite(arrays["betas"]).all(axis=1)
        & np.isfinite(arrays["transl"]).all(axis=1)
    )[0]
    if candidates.size == 0:
        raise RuntimeError(f"No valid MANO frames in {wilor_dir}")

    if args.frame_indices:
        requested_frames = [
            int(value.strip())
            for value in args.frame_indices.split(",")
            if value.strip()
        ]
        frame_to_out_idx = {
            int(frame_idx): int(out_idx)
            for out_idx, frame_idx in enumerate(arrays["frame_indices"].tolist())
        }
        missing = [
            frame_idx for frame_idx in requested_frames if frame_idx not in frame_to_out_idx
        ]
        if missing:
            raise ValueError(f"Requested video frame indices not found in MANO labels: {missing}")
        selected = np.asarray(
            [frame_to_out_idx[frame_idx] for frame_idx in requested_frames],
            dtype=np.int64,
        )
        selected = selected[np.isin(selected, candidates)]
        if selected.size == 0:
            raise RuntimeError("Requested frames exist but none are valid MANO frames")
    else:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(
            rng.choice(candidates, size=min(args.n_samples, candidates.size), replace=False)
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal = args.focal_length / args.image_size * max(image_w, image_h)
    sequential_frames: dict[int, np.ndarray] = {}
    if args.sequential_video_read:
        wanted_frames = {
            int(arrays["frame_indices"][int(out_idx)])
            for out_idx in selected.tolist()
        }
        max_frame = max(wanted_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(max_frame + 1):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx in wanted_frames:
                sequential_frames[frame_idx] = frame_bgr.copy()
                if len(sequential_frames) == len(wanted_frames):
                    break
        missing = sorted(wanted_frames - set(sequential_frames))
        if missing:
            raise RuntimeError(f"Sequential read missed requested frames: {missing}")

    device = torch.device(args.device)
    mano_model_path = args.mano_model_path
    if mano_model_path.is_dir() and (mano_model_path / "MANO_RIGHT.pkl").exists():
        mano_model_path = mano_model_path / "MANO_RIGHT.pkl"
    mano = smplx.MANO(
        model_path=str(mano_model_path),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)
    faces = mano.faces.astype(np.int64)
    detector = None
    if args.draw_yolo:
        if args.detector_path.exists():
            if YOLO is None:
                raise ImportError("ultralytics is required for --detector-path")
            detector = YOLO(str(args.detector_path))
        else:
            if WiLorHandPose3dEstimationPipeline is None:
                raise ImportError("wilor_mini is required when --detector-path is missing")
            pipeline_kwargs = {
                "device": device,
                "dtype": torch.float32,
            }
            if args.wilor_pretrained_dir:
                pipeline_kwargs["wilor_pretrained_dir"] = args.wilor_pretrained_dir
            detector = WiLorHandPose3dEstimationPipeline(**pipeline_kwargs).hand_detector

    items: list[dict[str, object]] = []
    for out_idx in selected.tolist():
        frame_idx = int(arrays["frame_indices"][out_idx])
        if args.sequential_video_read:
            frame_bgr = sequential_frames[frame_idx]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                print(f"skip out_idx={out_idx} frame={frame_idx}: video read failed")
                continue

        pose = arrays["pose"][out_idx]
        with torch.no_grad():
            mano_out = mano(
                global_orient=torch.from_numpy(pose[:3][None]).float().to(device),
                hand_pose=torch.from_numpy(pose[3:48][None]).float().to(device),
                betas=torch.from_numpy(arrays["betas"][out_idx][None]).float().to(device),
            )
        vertices_local = mano_out.vertices[0].detach().cpu().numpy().astype(np.float32)
        vertices_cam = vertices_local + arrays["transl"][out_idx][None, :]

        yolo_path = None
        yolo_detections: list[dict[str, object]] = []
        yolo_overlay = frame_bgr
        if detector is not None:
            yolo_overlay, yolo_detections = _draw_yolo_detections(
                frame_bgr,
                detector,
                args.target_hand,
                args.yolo_conf,
            )

        points, in_image = _project_vertices(vertices_cam, focal, image_w, image_h)
        depth_valid = vertices_cam[:, 2] > 1e-6
        overlay = _draw_wireframe(yolo_overlay, points, depth_valid, faces, (0, 180, 255))
        for point in points[in_image][::10]:
            cv2.circle(
                overlay,
                tuple(np.round(point).astype(np.int32)),
                1,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            f"out_idx={out_idx} frame={frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        prefix = f"frame_{frame_idx:06d}_out_{out_idx:06d}"
        raw_path = output_dir / f"{prefix}_raw.png"
        if detector is not None:
            yolo_path = output_dir / f"{prefix}_yolo.png"
        projection_path = output_dir / f"{prefix}_mesh_projection.png"
        glb_path = output_dir / f"{prefix}_mano_cam.glb"
        cv2.imwrite(str(raw_path), frame_bgr)
        if yolo_path is not None:
            cv2.imwrite(str(yolo_path), yolo_overlay)
        cv2.imwrite(str(projection_path), overlay)

        mesh = trimesh.Trimesh(vertices=vertices_cam, faces=faces, process=False)
        mesh.visual.vertex_colors = [255, 180, 0, 255]
        mesh.export(str(glb_path))

        item = {
            "out_idx": int(out_idx),
            "video_frame_index": frame_idx,
            "raw_png": str(raw_path),
            "yolo_png": str(yolo_path) if yolo_path is not None else None,
            "projection_png": str(projection_path),
            "mano_cam_glb": str(glb_path),
            "yolo_detections": yolo_detections,
            "num_projected_vertices_in_image": int(in_image.sum()),
            "depth_min": float(np.nanmin(vertices_cam[:, 2])),
            "depth_max": float(np.nanmax(vertices_cam[:, 2])),
        }
        items.append(item)
        print(
            f"wrote {prefix}: projected_vertices={item['num_projected_vertices_in_image']}"
        )

    cap.release()
    summary = {
        "session": str(session),
        "video_path": str(video_path),
        "seed": args.seed,
        "n_samples": len(items),
        "focal": focal,
        "image_w": image_w,
        "image_h": image_h,
        "items": items,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
