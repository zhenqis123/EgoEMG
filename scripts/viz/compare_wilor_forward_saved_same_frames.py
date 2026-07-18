#!/usr/bin/env python3
"""Compare live WiLoR forward output with saved WiLoR MANO labels on same frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
import torchvision.ops as tv_ops
import trimesh
from ultralytics import YOLO
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)
from wilor_mini.utils import utils as wilor_utils


def _read_frames_sequential(video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(frame_indices)
    max_frame = max(wanted)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: dict[int, np.ndarray] = {}
    for frame_idx in range(max_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            frames[frame_idx] = frame.copy()
            if len(frames) == len(wanted):
                break
    cap.release()
    missing = sorted(wanted - set(frames))
    if missing:
        raise RuntimeError(f"Sequential video read missed frames: {missing}")
    return frames


def _run_yolo_lowres(frame_rgb: np.ndarray, detector: YOLO, conf: float) -> list[dict]:
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
    for det in result:
        is_right = bool(int(det.boxes.cls.cpu().item()))
        box = det.boxes.data.cpu().squeeze().numpy()
        detections.append(
            {
                "hand": "right" if is_right else "left",
                "is_right": is_right,
                "score": float(box[4]) if box.shape[0] > 4 else float("nan"),
                "bbox_xyxy": [
                    float(box[0] * scale_x),
                    float(box[1] * scale_y),
                    float(box[2] * scale_x),
                    float(box[3] * scale_y),
                ],
            }
        )
    return detections


def _target_detection(detections: list[dict], target_hand: str) -> dict | None:
    target_is_right = target_hand == "right"
    for det in detections:
        if bool(det["is_right"]) == target_is_right:
            return det
    return None


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
        cv2.rectangle(
            out,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"{label} {float(det['score']):.2f}",
            (int(round(x1)), max(20, int(round(y1)) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _project_vertices(
    vertices_local: np.ndarray,
    translation: np.ndarray,
    focal: float,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_cam = vertices_local + translation[None, :]
    z = vertices_cam[:, 2]
    in_front = z > 1e-6
    points = np.empty((vertices_cam.shape[0], 2), dtype=np.float32)
    points[:, 0] = vertices_cam[:, 0] / z * focal + image_w / 2.0
    points[:, 1] = vertices_cam[:, 1] / z * focal + image_h / 2.0
    in_image = (
        in_front
        & (points[:, 0] >= 0)
        & (points[:, 0] < image_w)
        & (points[:, 1] >= 0)
        & (points[:, 1] < image_h)
    )
    return vertices_cam, points, in_image


def _draw_wireframe(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    in_front: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
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
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), color, 1, cv2.LINE_AA)
        cv2.line(out, tuple(pts[1]), tuple(pts[2]), color, 1, cv2.LINE_AA)
        cv2.line(out, tuple(pts[2]), tuple(pts[0]), color, 1, cv2.LINE_AA)
    return out


def _bbox_from_points(points: np.ndarray, mask: np.ndarray) -> list[float]:
    valid_points = points[mask]
    xy_min = valid_points.min(axis=0)
    xy_max = valid_points.max(axis=0)
    return [
        float((xy_min[0] + xy_max[0]) / 2.0),
        float((xy_min[1] + xy_max[1]) / 2.0),
        float(xy_max[0] - xy_min[0]),
        float(xy_max[1] - xy_min[1]),
    ]


def _forward_live(
    frame_rgb: np.ndarray,
    detection: dict,
    pipe: WiLorHandPose3dEstimationPipeline,
    target_hand: str,
) -> dict:
    target_is_right = target_hand == "right"
    box = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
    center = (box[2:4] + box[0:2]) / 2.0
    bbox_size = float((2.5 * (box[2:4] - box[0:2])).max())
    image_h, image_w = frame_rgb.shape[:2]
    image_size = np.array([image_w, image_h], dtype=np.float32)
    center_for_crop = center.copy()
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(
        device=pipe.device,
        dtype=torch.float32,
    )
    if not target_is_right:
        frame_tensor = torch.flip(frame_tensor, [2])
        center_for_crop[0] = image_w - center_for_crop[0] - 1
    half = bbox_size / 2.0
    roi = torch.tensor(
        [
            [
                0,
                center_for_crop[0] - half,
                center_for_crop[1] - half,
                center_for_crop[0] + half,
                center_for_crop[1] + half,
            ]
        ],
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
    output_np = {key: value.cpu().float().numpy() for key, value in output.items()}
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
    pred_cam[:, 1] = (1 if target_is_right else -1) * pred_cam[:, 1]
    focal = pipe.FOCAL_LENGTH / pipe.IMAGE_SIZE * image_size.max()
    translation = wilor_utils.cam_crop_to_full(
        pred_cam,
        center[None],
        bbox_size,
        image_size[None],
        focal,
    )[0]
    return {
        "vertices": vertices,
        "global_orient": global_orient.reshape(-1),
        "hand_pose": hand_pose.reshape(-1),
        "betas": output_np["betas"][0].copy(),
        "keypoints_3d": keypoints_3d,
        "pred_cam": pred_cam[0],
        "translation": translation,
        "focal": float(focal),
        "bbox_size": bbox_size,
        "center": center,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-indices", required=True)
    parser.add_argument("--target-hand", default="right", choices=["right", "left"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--yolo-conf", type=float, default=0.3)
    parser.add_argument("--wilor-pretrained-dir", default="data/pretrained_models")
    parser.add_argument(
        "--mano-model-path",
        type=Path,
        default=Path("data/pretrained_models/MANO_RIGHT.pkl"),
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_requested = [
        int(value.strip())
        for value in args.frame_indices.split(",")
        if value.strip()
    ]
    zed_dirs = sorted(
        path for path in args.session.iterdir() if path.is_dir() and path.name.startswith("ZED_")
    )
    video_path = zed_dirs[0] / "rgb.mkv"
    frames_bgr = _read_frames_sequential(video_path, frames_requested)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(
        device=torch.device(args.device),
        dtype=dtype,
        wilor_pretrained_dir=args.wilor_pretrained_dir,
    )
    detector = pipe.hand_detector

    wilor_dir = args.session / "wilor_mano"
    frame_indices = np.load(wilor_dir / "frame_indices.npy")
    frame_to_out_idx = {
        int(frame_idx): int(out_idx)
        for out_idx, frame_idx in enumerate(frame_indices.tolist())
    }
    valid = np.load(wilor_dir / "valid.npy").astype(bool)
    saved_betas = np.load(wilor_dir / "mano_betas.npy").astype(np.float32)
    saved_global_orient = np.load(wilor_dir / "mano_global_orient.npy").astype(np.float32)
    saved_hand_pose = np.load(wilor_dir / "mano_hand_pose.npy").astype(np.float32)
    saved_transl = np.load(wilor_dir / "mano_transl.npy").astype(np.float32)
    saved_pred_cam = np.load(wilor_dir / "pred_cam.npy").astype(np.float32)

    mano = smplx.MANO(
        model_path=str(args.mano_model_path),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(torch.device(args.device))
    faces = mano.faces.astype(np.int64)

    summary: dict[str, object] = {
        "session": str(args.session),
        "video_path": str(video_path),
        "frame_indices": frames_requested,
        "target_hand": args.target_hand,
        "device": args.device,
        "dtype": args.dtype,
        "items": [],
    }
    for frame_idx in frames_requested:
        if frame_idx not in frame_to_out_idx:
            raise ValueError(f"frame {frame_idx} not found in saved labels")
        out_idx = frame_to_out_idx[frame_idx]
        if not valid[out_idx]:
            raise ValueError(f"frame {frame_idx} exists but saved label is invalid")

        frame_bgr = frames_bgr[frame_idx]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_h, image_w = frame_bgr.shape[:2]
        detections = _run_yolo_lowres(frame_rgb, detector, args.yolo_conf)
        detection = _target_detection(detections, args.target_hand)
        if detection is None:
            raise RuntimeError(f"frame {frame_idx}: no target-hand detection")

        live = _forward_live(frame_rgb, detection, pipe, args.target_hand)
        pose = np.concatenate([saved_global_orient[out_idx], saved_hand_pose[out_idx]], axis=0)
        with torch.no_grad():
            mano_out = mano(
                global_orient=torch.from_numpy(pose[:3][None]).float().to(torch.device(args.device)),
                hand_pose=torch.from_numpy(pose[3:48][None]).float().to(torch.device(args.device)),
                betas=torch.from_numpy(saved_betas[out_idx][None]).float().to(torch.device(args.device)),
            )
        saved_vertices = mano_out.vertices[0].detach().cpu().numpy().astype(np.float32)

        live_vertices_cam, live_points, live_in_image = _project_vertices(
            live["vertices"], live["translation"], live["focal"], image_w, image_h
        )
        saved_vertices_cam, saved_points, saved_in_image = _project_vertices(
            saved_vertices, saved_transl[out_idx], live["focal"], image_w, image_h
        )

        yolo_overlay = _draw_detections(frame_bgr, detections, args.target_hand)
        live_overlay = _draw_wireframe(
            yolo_overlay,
            live_points,
            live_vertices_cam[:, 2] > 1e-6,
            faces,
            (0, 180, 255),
        )
        saved_overlay = _draw_wireframe(
            yolo_overlay,
            saved_points,
            saved_vertices_cam[:, 2] > 1e-6,
            faces,
            (255, 0, 255),
        )
        diff = np.abs(live_overlay.astype(np.int16) - saved_overlay.astype(np.int16)).astype(np.uint8)
        canvas = np.concatenate([yolo_overlay, live_overlay, saved_overlay, diff], axis=1)

        prefix = f"frame_{frame_idx:06d}_out_{out_idx:06d}"
        raw_path = output_dir / f"{prefix}_raw.png"
        yolo_path = output_dir / f"{prefix}_yolo.png"
        live_path = output_dir / f"{prefix}_live_forward.png"
        saved_path = output_dir / f"{prefix}_saved_label.png"
        canvas_path = output_dir / f"{prefix}_compare.png"
        cv2.imwrite(str(raw_path), frame_bgr)
        cv2.imwrite(str(yolo_path), yolo_overlay)
        cv2.imwrite(str(live_path), live_overlay)
        cv2.imwrite(str(saved_path), saved_overlay)
        cv2.imwrite(str(canvas_path), canvas)

        live_glb = output_dir / f"{prefix}_live_forward.glb"
        saved_glb = output_dir / f"{prefix}_saved_label.glb"
        for path, vertices_cam, color in [
            (live_glb, live_vertices_cam, [255, 180, 0, 255]),
            (saved_glb, saved_vertices_cam, [255, 0, 255, 255]),
        ]:
            mesh = trimesh.Trimesh(vertices=vertices_cam, faces=faces, process=False)
            mesh.visual.vertex_colors = color
            mesh.export(str(path))

        live_box = _bbox_from_points(live_points, live_in_image)
        saved_box = _bbox_from_points(saved_points, saved_in_image)
        item = {
            "frame_idx": frame_idx,
            "out_idx": out_idx,
            "raw_png": str(raw_path),
            "yolo_png": str(yolo_path),
            "live_forward_png": str(live_path),
            "saved_label_png": str(saved_path),
            "compare_png": str(canvas_path),
            "live_forward_glb": str(live_glb),
            "saved_label_glb": str(saved_glb),
            "yolo_detections": detections,
            "live_pred_cam": live["pred_cam"].tolist(),
            "saved_pred_cam": saved_pred_cam[out_idx].tolist(),
            "pred_cam_l2": float(np.linalg.norm(live["pred_cam"] - saved_pred_cam[out_idx])),
            "live_translation": live["translation"].tolist(),
            "saved_translation": saved_transl[out_idx].tolist(),
            "translation_l2": float(np.linalg.norm(live["translation"] - saved_transl[out_idx])),
            "global_orient_l2": float(
                np.linalg.norm(live["global_orient"] - saved_global_orient[out_idx])
            ),
            "hand_pose_l2": float(np.linalg.norm(live["hand_pose"] - saved_hand_pose[out_idx])),
            "betas_l2": float(np.linalg.norm(live["betas"] - saved_betas[out_idx])),
            "live_projected_vertices": int(live_in_image.sum()),
            "saved_projected_vertices": int(saved_in_image.sum()),
            "live_projection_bbox_cxcywh": live_box,
            "saved_projection_bbox_cxcywh": saved_box,
            "projection_bbox_center_l2": float(
                np.linalg.norm(np.asarray(live_box[:2]) - np.asarray(saved_box[:2]))
            ),
            "projection_bbox_size_l2": float(
                np.linalg.norm(np.asarray(live_box[2:]) - np.asarray(saved_box[2:]))
            ),
        }
        summary["items"].append(item)
        print(
            f"frame={frame_idx} out={out_idx} pred_cam_l2={item['pred_cam_l2']:.4f} "
            f"trans_l2={item['translation_l2']:.4f} bbox_center_l2={item['projection_bbox_center_l2']:.2f}"
        )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
