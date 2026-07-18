#!/usr/bin/env python3
"""Re-run YOLO+WiLoR on random session frames and save mesh projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops
import trimesh

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)
from wilor_mini.utils import utils as wilor_utils


def _sample_frame_indices(
    session: Path,
    video_count: int,
    n_samples: int,
    seed: int,
    prefer_existing_valid: bool,
) -> np.ndarray:
    if prefer_existing_valid:
        valid_path = session / "wilor_mano" / "valid.npy"
        frame_idx_path = session / "wilor_mano" / "frame_indices.npy"
        if valid_path.exists() and frame_idx_path.exists():
            valid = np.load(valid_path)
            frame_indices = np.load(frame_idx_path)
            candidates = frame_indices[valid.astype(bool)]
            candidates = candidates[(candidates >= 0) & (candidates < video_count)]
        else:
            candidates = np.arange(video_count, dtype=np.int64)
    else:
        candidates = np.arange(video_count, dtype=np.int64)
    if candidates.size == 0:
        raise RuntimeError("No candidate frames to sample")
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=min(n_samples, candidates.size), replace=False)
    return np.sort(selected.astype(np.int64))


def _run_yolo_lowres(frame_rgb: np.ndarray, detector, conf: float) -> tuple[list[dict], list[np.ndarray]]:
    image_h, image_w = frame_rgb.shape[:2]
    yolo_h = 256
    yolo_w = max(int(image_w * yolo_h / image_h), 1)
    scale_x = image_w / float(yolo_w)
    scale_y = image_h / float(yolo_h)
    frame_yolo = cv2.resize(frame_rgb, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)
    result = detector(frame_yolo, conf=conf, verbose=False)
    if isinstance(result, list):
        result = result[0]

    detections: list[dict] = []
    boxes_full: list[np.ndarray] = []
    for det in result:
        is_right = bool(int(det.boxes.cls.cpu().item()))
        box = det.boxes.data.cpu().squeeze().numpy()
        full = np.array(
            [
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y,
            ],
            dtype=np.float32,
        )
        score = float(box[4]) if box.shape[0] > 4 else float("nan")
        boxes_full.append(full)
        detections.append(
            {
                "hand": "right" if is_right else "left",
                "is_right": is_right,
                "score": score,
                "bbox_xyxy": full.tolist(),
            }
        )
    return detections, boxes_full


def _draw_detections(frame_bgr: np.ndarray, detections: list[dict], target_hand: str) -> np.ndarray:
    out = frame_bgr.copy()
    target_is_right = target_hand == "right"
    for det in detections:
        is_right = bool(det["is_right"])
        x1, y1, x2, y2 = det["bbox_xyxy"]
        color = (0, 255, 0) if is_right else (255, 0, 0)
        label = "YOLO R" if is_right else "YOLO L"
        if is_right == target_is_right:
            label += " TARGET"
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, pt1, pt2, color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"{label} {float(det['score']):.2f}",
            (pt1[0], max(20, pt1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _project_vertices(
    vertices: np.ndarray,
    translation: np.ndarray,
    focal: float,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices_cam = vertices + translation[None, :]
    z = vertices_cam[:, 2]
    valid = z > 1e-6
    points = np.empty((vertices_cam.shape[0], 2), dtype=np.float32)
    points[:, 0] = vertices_cam[:, 0] / z * focal + image_w / 2.0
    points[:, 1] = vertices_cam[:, 1] / z * focal + image_h / 2.0
    in_image = (
        valid
        & (points[:, 0] >= 0)
        & (points[:, 0] < image_w)
        & (points[:, 1] >= 0)
        & (points[:, 1] < image_h)
    )
    return points, in_image


def _draw_wireframe(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    in_front: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    out = frame_bgr.copy()
    image_h, image_w = out.shape[:2]
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (in_front[i0] and in_front[i1] and in_front[i2]):
            continue
        pts = np.round(points[[i0, i1, i2]]).astype(np.int32)
        if (
            (pts[:, 0] < -1000).any()
            or (pts[:, 0] > image_w + 1000).any()
            or (pts[:, 1] < -1000).any()
            or (pts[:, 1] > image_h + 1000).any()
        ):
            continue
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), (0, 180, 255), 1, cv2.LINE_AA)
        cv2.line(out, tuple(pts[1]), tuple(pts[2]), (0, 180, 255), 1, cv2.LINE_AA)
        cv2.line(out, tuple(pts[2]), tuple(pts[0]), (0, 180, 255), 1, cv2.LINE_AA)
    return out


def _forward_target_hand(
    frame_rgb: np.ndarray,
    boxes_full: list[np.ndarray],
    detections: list[dict],
    target_hand: str,
    pipe: WiLorHandPose3dEstimationPipeline,
) -> dict | None:
    target_is_right = target_hand == "right"
    selected: tuple[np.ndarray, dict] | None = None
    for box, det in zip(boxes_full, detections):
        if bool(det["is_right"]) == target_is_right:
            selected = (box, det)
            break
    if selected is None:
        return None

    box, det = selected
    center = (box[2:4] + box[0:2]) / 2.0
    scale = 2.5 * (box[2:4] - box[0:2])
    bbox_size = float(scale.max())
    image_h, image_w = frame_rgb.shape[:2]
    image_size = np.array([image_w, image_h], dtype=np.float32)
    flip = not target_is_right

    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(
        device=pipe.device,
        dtype=torch.float32,
    )
    if flip:
        frame_tensor = torch.flip(frame_tensor, [2])
        center_for_crop = center.copy()
        center_for_crop[0] = image_w - center_for_crop[0] - 1
    else:
        center_for_crop = center
    half = bbox_size / 2.0
    roi = torch.tensor(
        [[0, center_for_crop[0] - half, center_for_crop[1] - half, center_for_crop[0] + half, center_for_crop[1] + half]],
        device=pipe.device,
        dtype=torch.float32,
    )
    patch = tv_ops.roi_align(
        frame_tensor[None],
        roi,
        output_size=(pipe.IMAGE_SIZE, pipe.IMAGE_SIZE),
        spatial_scale=1.0,
        aligned=True,
    )
    patch = patch.permute(0, 2, 3, 1).to(dtype=pipe.dtype)
    with torch.no_grad():
        output = pipe.wilor_model(patch)
    output_np = {k: v.cpu().float().numpy() for k, v in output.items()}

    pred_cam = output_np["pred_cam"].copy()
    vertices = output_np["pred_vertices"][0].copy()
    keypoints_3d = output_np["pred_keypoints_3d"][0].copy()
    global_orient = output_np["global_orient"].copy()
    hand_pose = output_np["hand_pose"].copy()
    if not target_is_right:
        vertices[:, 0] = -vertices[:, 0]
        keypoints_3d[:, 0] = -keypoints_3d[:, 0]
        global_orient[:, :, 1:3] = -global_orient[:, :, 1:3]
        hand_pose[:, :, 1:3] = -hand_pose[:, :, 1:3]
    multiplier = 1 if target_is_right else -1
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    scaled_focal = pipe.FOCAL_LENGTH / pipe.IMAGE_SIZE * image_size.max()
    pred_cam_t_full = wilor_utils.cam_crop_to_full(
        pred_cam,
        center[None],
        bbox_size,
        image_size[None],
        scaled_focal,
    )
    return {
        "detection": det,
        "bbox_size": bbox_size,
        "center": center.tolist(),
        "vertices": vertices,
        "keypoints_3d": keypoints_3d,
        "pred_cam": pred_cam[0],
        "pred_cam_t_full": pred_cam_t_full[0],
        "scaled_focal_length": float(scaled_focal),
        "global_orient": global_orient.reshape(-1).tolist(),
        "hand_pose": hand_pose.reshape(-1).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--wilor-pretrained-dir", type=Path, default=Path("data/pretrained_models"))
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=140914)
    parser.add_argument("--target-hand", default="right", choices=["right", "left"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--yolo-conf", type=float, default=0.3)
    parser.add_argument("--prefer-existing-valid", action="store_true", default=True)
    args = parser.parse_args()

    session = args.session
    output_dir = args.output or session / "wilor_mano" / "random_forward_viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    zed_dirs = sorted(p for p in session.iterdir() if p.is_dir() and p.name.startswith("ZED_"))
    if not zed_dirs:
        raise FileNotFoundError(f"No ZED_* directory under {session}")
    video_path = zed_dirs[0] / "rgb.mkv"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = _sample_frame_indices(
        session,
        video_count,
        args.n_samples,
        args.seed,
        args.prefer_existing_valid,
    )

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.device == "cpu":
        dtype = torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(
        device=torch.device(args.device),
        dtype=dtype,
        wilor_pretrained_dir=str(args.wilor_pretrained_dir),
        verbose=False,
    )
    faces = pipe.wilor_model.mano.faces.astype(np.int64)

    items: list[dict[str, object]] = []
    for frame_idx in frame_indices.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections, boxes_full = _run_yolo_lowres(frame_rgb, pipe.hand_detector, args.yolo_conf)
        yolo_overlay = _draw_detections(frame_bgr, detections, args.target_hand)
        forward = _forward_target_hand(
            frame_rgb,
            boxes_full,
            detections,
            args.target_hand,
            pipe,
        )

        prefix = f"frame_{int(frame_idx):06d}"
        raw_path = output_dir / f"{prefix}_raw.png"
        yolo_path = output_dir / f"{prefix}_yolo.png"
        cv2.imwrite(str(raw_path), frame_bgr)
        cv2.imwrite(str(yolo_path), yolo_overlay)
        item: dict[str, object] = {
            "video_frame_index": int(frame_idx),
            "raw_png": str(raw_path),
            "yolo_png": str(yolo_path),
            "yolo_detections": detections,
            "target_forward_valid": forward is not None,
        }
        if forward is not None:
            vertices = forward["vertices"]
            pred_cam_t_full = forward["pred_cam_t_full"]
            focal = float(forward["scaled_focal_length"])
            points, in_image = _project_vertices(
                vertices,
                pred_cam_t_full,
                focal,
                frame_bgr.shape[1],
                frame_bgr.shape[0],
            )
            in_front = (vertices[:, 2] + pred_cam_t_full[2]) > 1e-6
            projection = _draw_wireframe(yolo_overlay, points, in_front, faces)
            for point in points[in_image][::10]:
                cv2.circle(
                    projection,
                    tuple(np.round(point).astype(np.int32)),
                    1,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                projection,
                f"rerun frame={int(frame_idx)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            projection_path = output_dir / f"{prefix}_mesh_projection.png"
            glb_path = output_dir / f"{prefix}_mano_cam.glb"
            cv2.imwrite(str(projection_path), projection)
            mesh = trimesh.Trimesh(
                vertices=vertices + pred_cam_t_full[None, :],
                faces=faces,
                process=False,
            )
            mesh.visual.vertex_colors = [255, 180, 0, 255]
            mesh.export(str(glb_path))
            item.update(
                {
                    "projection_png": str(projection_path),
                    "mano_cam_glb": str(glb_path),
                    "target_detection": forward["detection"],
                    "center": forward["center"],
                    "bbox_size": forward["bbox_size"],
                    "pred_cam": np.asarray(forward["pred_cam"]).tolist(),
                    "pred_cam_t_full": np.asarray(pred_cam_t_full).tolist(),
                    "scaled_focal_length": focal,
                    "num_projected_vertices_in_image": int(in_image.sum()),
                }
            )
        items.append(item)
        print(
            f"frame={int(frame_idx)} dets={len(detections)} "
            f"target={item['target_forward_valid']}"
        )

    cap.release()
    summary = {
        "session": str(session),
        "video_path": str(video_path),
        "seed": args.seed,
        "n_samples": len(items),
        "target_hand": args.target_hand,
        "wilor_pretrained_dir": str(args.wilor_pretrained_dir),
        "items": items,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
