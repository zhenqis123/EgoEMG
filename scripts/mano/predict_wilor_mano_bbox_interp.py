#!/usr/bin/env python3
"""Multi-GPU WiLoR MANO prediction with interpolated YOLO bboxes for missing frames.

Uses decord for fast sequential video decoding and torch.multiprocessing.spawn
to distribute YOLO detection + WiLoR inference across all available GPUs.

Workflow (per session):
  1. All GPUs run YOLO detection on their chunk of frames in parallel.
  2. Rank 0 collects chunk results, interpolates missing bboxes, broadcasts.
  3. All GPUs run WiLoR forward on their chunk using the interpolated bboxes.
  4. Rank 0 collects MANO results and writes final .npy output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tempfile

# decord requires a newer libstdc++ than the system provides
if "LD_LIBRARY_PATH" not in os.environ:
    _conda_lib = os.path.join(sys.prefix, "lib")
    if os.path.isdir(_conda_lib):
        os.environ["LD_LIBRARY_PATH"] = _conda_lib
    else:
        os.environ["LD_LIBRARY_PATH"] = os.path.join(os.path.dirname(sys.prefix), "lib")
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops
from tqdm import tqdm
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)
from wilor_mini.utils import utils as wilor_utils


@dataclass
class FrameMeta:
    out_idx: int
    frame_idx: int
    center: np.ndarray
    bbox_size: float
    img_size: np.ndarray
    bbox_source: int


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_color_timestamps(timestamp_csv: Path) -> tuple[list[int], list[int]]:
    frame_ids, timestamps_us = [], []
    with open(timestamp_csv) as f:
        for row in csv.DictReader(f):
            if row["stream_name"] == "color":
                frame_ids.append(int(row["frame_id"]))
                timestamps_us.append(int(row["align_timestamp_us"]))
    return frame_ids, timestamps_us


def find_zed_dir(session_dir: Path) -> Path:
    for directory in session_dir.iterdir():
        if directory.is_dir() and directory.name.startswith("ZED_"):
            return directory
    raise FileNotFoundError(f"No ZED_* directory in {session_dir}")


def _timestamp_at(timestamps_us: list[int], frame_idx: int) -> int:
    if 0 <= frame_idx < len(timestamps_us):
        return int(timestamps_us[frame_idx])
    return 0


# ── bbox detection ───────────────────────────────────────────────────────────

def _detect_target_bboxes(
    detector,
    frames_rgb: list[np.ndarray],
    target_is_right: bool,
    yolo_conf: float,
    yolo_input_height: int,
) -> tuple[list[tuple[np.ndarray, float] | None], int, int]:
    if not frames_rgb:
        return [], 0, 0

    yolo_images = []
    scales: list[tuple[float, float]] = []
    for frame in frames_rgb:
        image_h, image_w = frame.shape[:2]
        if yolo_input_height > 0:
            yolo_h = yolo_input_height
            yolo_w = max(int(image_w * yolo_h / image_h), 1)
            scales.append((image_w / float(yolo_w), image_h / float(yolo_h)))
            yolo_images.append(cv2.resize(frame, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR))
        else:
            scales.append((1.0, 1.0))
            yolo_images.append(frame)

    results = detector(yolo_images, conf=yolo_conf, verbose=False)
    if not isinstance(results, list):
        results = [results]

    found: list[tuple[np.ndarray, float] | None] = []
    no_det = 0
    wrong_hand = 0
    for result, (scale_x, scale_y) in zip(results, scales):
        if result is None or len(result) == 0:
            found.append(None)
            no_det += 1
            continue
        target: tuple[np.ndarray, float] | None = None
        for det in result:
            is_right = bool(int(det.boxes.cls.cpu().item()))
            if is_right != target_is_right:
                continue
            box = det.boxes.data.cpu().squeeze().numpy()[:4]
            x1 = float(box[0] * scale_x)
            y1 = float(box[1] * scale_y)
            x2 = float(box[2] * scale_x)
            y2 = float(box[3] * scale_y)
            center = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            bbox_size = float(max(x2 - x1, y2 - y1) * 2.5)
            target = (center, bbox_size)
            break
        if target is None:
            wrong_hand += 1
        found.append(target)
    return found, no_det, wrong_hand


def _interpolate_bboxes(
    centers: np.ndarray,
    bbox_sizes: np.ndarray,
    detected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = len(detected)
    valid_idx = np.where(detected)[0]
    if valid_idx.size == 0:
        raise RuntimeError("No target-hand bboxes detected; cannot interpolate.")
    all_idx = np.arange(n_frames, dtype=np.float32)
    centers_out = centers.copy()
    sizes_out = bbox_sizes.copy()
    for dim in range(2):
        centers_out[:, dim] = np.interp(
            all_idx, valid_idx.astype(np.float32), centers[valid_idx, dim],
        ).astype(np.float32)
    sizes_out[:] = np.interp(
        all_idx, valid_idx.astype(np.float32), bbox_sizes[valid_idx],
    ).astype(np.float32)
    bbox_source = np.zeros(n_frames, dtype=np.uint8)
    bbox_source[~detected] = 1
    bbox_source[: valid_idx[0]] = 2
    bbox_source[valid_idx[-1] + 1:] = 2
    return centers_out, sizes_out, bbox_source


# ── single-GPU worker ────────────────────────────────────────────────────────

def _worker_yolo(
    rank: int,
    world_size: int,
    video_path: Path,
    n_out: int,
    timestamps_us: list[int],
    target_is_right: bool,
    yolo_conf: float,
    yolo_input_height: int,
    frame_batch: int,
    wilor_pretrained_dir: str | None,
    device_str: str,
    dtype_str: str,
    tmp_dir: Path,
) -> None:
    """Worker: run YOLO detection on this rank's chunk, save bboxes to tmp_dir."""
    device = torch.device(f"cuda:{rank}")
    dtype = getattr(torch, dtype_str)
    kwargs: dict = {"device": device, "dtype": dtype}
    if wilor_pretrained_dir:
        kwargs["wilor_pretrained_dir"] = wilor_pretrained_dir
    pipe = WiLorHandPose3dEstimationPipeline(**kwargs)
    detector = pipe.hand_detector

    # Chunk boundaries
    chunk_start = (n_out * rank) // world_size
    chunk_end = (n_out * (rank + 1)) // world_size

    import decord
    vr = decord.VideoReader(str(video_path))

    centers_chunk = np.full((chunk_end - chunk_start, 2), np.nan, dtype=np.float32)
    bbox_sizes_chunk = np.full(chunk_end - chunk_start, np.nan, dtype=np.float32)
    bbox_detected_chunk = np.zeros(chunk_end - chunk_start, dtype=bool)
    n_no_det = 0
    n_wrong_hand = 0

    pbar = tqdm(total=chunk_end - chunk_start, desc=f"YOLO GPU{rank}", unit="f",
                position=rank)
    for batch_start in range(chunk_start, chunk_end, frame_batch):
        batch_end = min(batch_start + frame_batch, chunk_end)
        batch_indices = list(range(batch_start, batch_end))
        try:
            frames = vr.get_batch(batch_indices).asnumpy()
        except Exception:
            pbar.update(len(batch_indices))
            continue
        frames_rgb = [frames[i] for i in range(len(frames))]
        detected, no_det, wrong_hand = _detect_target_bboxes(
            detector, frames_rgb, target_is_right, yolo_conf, yolo_input_height,
        )
        n_no_det += no_det
        n_wrong_hand += wrong_hand
        for j, target in enumerate(detected):
            local_idx = batch_indices[j] - chunk_start
            if target is None:
                continue
            center, bbox_size = target
            centers_chunk[local_idx] = center
            bbox_sizes_chunk[local_idx] = bbox_size
            bbox_detected_chunk[local_idx] = True
        pbar.update(len(batch_indices))
    pbar.close()

    # Save chunk results
    chunk_file = tmp_dir / f"bbox_chunk_{rank}.npz"
    np.savez(chunk_file,
             centers=centers_chunk, bbox_sizes=bbox_sizes_chunk,
             bbox_detected=bbox_detected_chunk,
             n_no_det=n_no_det, n_wrong_hand=n_wrong_hand,
             chunk_start=chunk_start, chunk_end=chunk_end)
    print(f"[GPU{rank}] YOLO done: detected={bbox_detected_chunk.sum()}, "
          f"no_det={n_no_det}, wrong_hand={n_wrong_hand}", flush=True)


def _worker_wilor(
    rank: int,
    world_size: int,
    video_path: Path,
    n_out: int,
    timestamps_us: list[int],
    target_is_right: bool,
    frame_batch: int,
    wilor_pretrained_dir: str | None,
    device_str: str,
    dtype_str: str,
    tmp_dir: Path,
) -> None:
    """Worker: run WiLoR forward on this rank's chunk using interpolated bboxes."""
    device = torch.device(f"cuda:{rank}")
    dtype = getattr(torch, dtype_str)
    kwargs: dict = {"device": device, "dtype": dtype}
    if wilor_pretrained_dir:
        kwargs["wilor_pretrained_dir"] = wilor_pretrained_dir
    pipe = WiLorHandPose3dEstimationPipeline(**kwargs)
    model = pipe.wilor_model

    chunk_start = (n_out * rank) // world_size
    chunk_end = (n_out * (rank + 1)) // world_size
    chunk_len = chunk_end - chunk_start

    # Load interpolated bboxes for this chunk
    bbox_data = np.load(tmp_dir / "bboxes_interp.npz")
    centers = bbox_data["centers"][chunk_start:chunk_end]
    bbox_sizes = bbox_data["bbox_sizes"][chunk_start:chunk_end]
    bbox_source = bbox_data["bbox_source"][chunk_start:chunk_end]

    import decord
    vr = decord.VideoReader(str(video_path))
    image_w = int(vr[0].shape[1])
    image_h = int(vr[0].shape[0])
    image_size = np.asarray([image_w, image_h], dtype=np.float32)

    betas_buf = np.full((chunk_len, 10), np.nan, dtype=np.float32)
    global_orient_buf = np.full((chunk_len, 3), np.nan, dtype=np.float32)
    hand_pose_buf = np.full((chunk_len, 45), np.nan, dtype=np.float32)
    transl_buf = np.full((chunk_len, 3), np.nan, dtype=np.float32)
    keypoints_3d_buf = np.full((chunk_len, 21, 3), np.nan, dtype=np.float32)
    pred_cam_buf = np.full((chunk_len, 3), np.nan, dtype=np.float32)
    valid_buf = np.zeros(chunk_len, dtype=bool)

    pbar = tqdm(total=chunk_len, desc=f"WiLoR GPU{rank}", unit="f", position=rank)
    for batch_start in range(chunk_start, chunk_end, frame_batch):
        batch_end = min(batch_start + frame_batch, chunk_end)
        batch_indices = list(range(batch_start, batch_end))
        try:
            frames = vr.get_batch(batch_indices).asnumpy()
        except Exception:
            pbar.update(len(batch_indices))
            continue
        frames_rgb = [frames[i] for i in range(len(frames))]
        if not frames_rgb:
            pbar.update(len(batch_indices))
            continue

        frame_tensors = [
            torch.from_numpy(f).permute(2, 0, 1).to(device=device, dtype=torch.float32)
            for f in frames_rgb
        ]
        frames_batch = torch.stack(frame_tensors)
        roi_boxes = []
        metas: list[FrameMeta] = []
        for j, global_idx in enumerate(batch_indices):
            local_offset = global_idx - chunk_start
            center = centers[local_offset]
            half = float(bbox_sizes[local_offset]) / 2.0
            roi_boxes.append([
                j,
                float(center[0] - half),
                float(center[1] - half),
                float(center[0] + half),
                float(center[1] + half),
            ])
            metas.append(FrameMeta(
                out_idx=local_offset,
                frame_idx=global_idx,
                center=center,
                bbox_size=float(bbox_sizes[local_offset]),
                img_size=image_size,
                bbox_source=int(bbox_source[local_offset]),
            ))

        boxes_t = torch.tensor(roi_boxes, device=device, dtype=torch.float32)
        patches_chw = tv_ops.roi_align(
            frames_batch, boxes_t,
            output_size=(pipe.IMAGE_SIZE, pipe.IMAGE_SIZE),
            spatial_scale=1.0, aligned=True,
        )
        patches_nhwc = patches_chw.permute(0, 2, 3, 1).to(dtype=dtype)
        with torch.no_grad():
            outputs = model(patches_nhwc)
        outputs = {key: value.cpu().float().numpy() for key, value in outputs.items()}

        for idx, meta in enumerate(metas):
            pred_cam = outputs["pred_cam"][idx].copy()
            pred_cam[1] = (1 if target_is_right else -1) * pred_cam[1]
            global_or = outputs["global_orient"][idx].copy()
            hand_pose = outputs["hand_pose"][idx].copy()
            keypoints_3d = outputs["pred_keypoints_3d"][idx].copy()
            if not target_is_right:
                keypoints_3d[:, 0] = -keypoints_3d[:, 0]
                global_or[:, 1:3] = -global_or[:, 1:3]
                hand_pose[:, 1:3] = -hand_pose[:, 1:3]
            scaled_focal = pipe.FOCAL_LENGTH / pipe.IMAGE_SIZE * meta.img_size.max()
            translation = wilor_utils.cam_crop_to_full(
                pred_cam[None], meta.center[None], meta.bbox_size,
                meta.img_size[None], scaled_focal,
            )[0]

            out_i = meta.out_idx
            betas_buf[out_i] = outputs["betas"][idx]
            global_orient_buf[out_i] = global_or.reshape(-1)
            hand_pose_buf[out_i] = hand_pose.reshape(-1)
            transl_buf[out_i] = translation
            keypoints_3d_buf[out_i] = keypoints_3d
            pred_cam_buf[out_i] = pred_cam
            valid_buf[out_i] = True
        pbar.update(len(batch_indices))
    pbar.close()

    chunk_file = tmp_dir / f"mano_chunk_{rank}.npz"
    np.savez(chunk_file,
             betas=betas_buf, global_orient=global_orient_buf,
             hand_pose=hand_pose_buf, transl=transl_buf,
             keypoints_3d=keypoints_3d_buf, pred_cam=pred_cam_buf,
             valid=valid_buf)
    print(f"[GPU{rank}] WiLoR done: valid={valid_buf.sum()}/{chunk_len}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def run_pipeline(
    session_dir: Path,
    hand: str,
    devices: list[int],
    dtype_str: str,
    wilor_pretrained_dir: str | None,
    frame_batch: int,
    yolo_conf: float,
    yolo_input_height: int,
) -> None:
    zed_dir = find_zed_dir(session_dir)
    video_path = zed_dir / "rgb.mkv"
    timestamps_csv = zed_dir / "timestamps.csv"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not timestamps_csv.exists():
        raise FileNotFoundError(timestamps_csv)

    _, timestamps_us = parse_color_timestamps(timestamps_csv)

    import decord
    vr = decord.VideoReader(str(video_path))
    n_frames_video = len(vr)
    n_out = min(n_frames_video, len(timestamps_us))
    fps = vr.get_avg_fps()
    del vr

    target_is_right = hand == "right"
    world_size = len(devices)
    device_str = ",".join(str(d) for d in devices)

    print(f"Session={session_dir} frames={n_out} fps={fps:.1f} "
          f"GPUs={device_str} dtype={dtype_str}")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # ── Stage 1: Multi-GPU YOLO detection ──
        print("\n=== Stage 1: YOLO bbox detection (multi-GPU) ===")
        t0 = time.time()
        if world_size == 1:
            _worker_yolo(0, 1, video_path, n_out, timestamps_us,
                         target_is_right, yolo_conf, yolo_input_height,
                         frame_batch, wilor_pretrained_dir, "cuda:0", dtype_str, tmp_dir)
        else:
            ctx = torch.multiprocessing.get_context("spawn")
            processes = []
            for rank, dev_id in enumerate(devices):
                p = ctx.Process(
                    target=_worker_yolo,
                    args=(rank, world_size, video_path, n_out, timestamps_us,
                          target_is_right, yolo_conf, yolo_input_height,
                          frame_batch, wilor_pretrained_dir,
                          f"cuda:{dev_id}", dtype_str, tmp_dir),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        print(f"YOLO stage done in {time.time() - t0:.1f}s")

        # ── Merge & interpolate bboxes ──
        print("\n=== Interpolating bboxes ===")
        centers = np.full((n_out, 2), np.nan, dtype=np.float32)
        bbox_sizes = np.full(n_out, np.nan, dtype=np.float32)
        bbox_detected = np.zeros(n_out, dtype=bool)
        total_no_det = 0
        total_wrong_hand = 0
        for rank in range(world_size):
            chunk = np.load(tmp_dir / f"bbox_chunk_{rank}.npz")
            cs = int(chunk["chunk_start"])
            ce = int(chunk["chunk_end"])
            centers[cs:ce] = chunk["centers"]
            bbox_sizes[cs:ce] = chunk["bbox_sizes"]
            bbox_detected[cs:ce] = chunk["bbox_detected"]
            total_no_det += int(chunk["n_no_det"])
            total_wrong_hand += int(chunk["n_wrong_hand"])

        centers, bbox_sizes, bbox_source = _interpolate_bboxes(
            centers, bbox_sizes, bbox_detected)
        n_bbox_detected = int(bbox_detected.sum())
        n_bbox_interpolated = int((bbox_source == 1).sum())
        n_bbox_edge_filled = int((bbox_source == 2).sum())
        bbox_det_rate = n_bbox_detected / n_out * 100
        print(f"BBox detected={n_bbox_detected} ({bbox_det_rate:.1f}%), "
              f"interpolated={n_bbox_interpolated}, edge_filled={n_bbox_edge_filled}")
        if bbox_det_rate < 5.0:
            print(f"WARNING: Very low detection rate ({bbox_det_rate:.1f}%). "
                  f"Interpolated bboxes may be unreliable.")

        np.savez(tmp_dir / "bboxes_interp.npz",
                 centers=centers, bbox_sizes=bbox_sizes, bbox_source=bbox_source)

        # ── Stage 2: Multi-GPU WiLoR forward ──
        print(f"\n=== Stage 2: WiLoR forward (multi-GPU) ===")
        t0 = time.time()
        if world_size == 1:
            _worker_wilor(0, 1, video_path, n_out, timestamps_us,
                          target_is_right, frame_batch, wilor_pretrained_dir,
                          "cuda:0", dtype_str, tmp_dir)
        else:
            ctx = torch.multiprocessing.get_context("spawn")
            processes = []
            for rank, dev_id in enumerate(devices):
                p = ctx.Process(
                    target=_worker_wilor,
                    args=(rank, world_size, video_path, n_out, timestamps_us,
                          target_is_right, frame_batch, wilor_pretrained_dir,
                          f"cuda:{dev_id}", dtype_str, tmp_dir),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        print(f"WiLoR stage done in {time.time() - t0:.1f}s")

        # ── Merge MANO results ──
        print("\n=== Merging results ===")
        betas_buf = np.full((n_out, 10), np.nan, dtype=np.float32)
        global_orient_buf = np.full((n_out, 3), np.nan, dtype=np.float32)
        hand_pose_buf = np.full((n_out, 45), np.nan, dtype=np.float32)
        transl_buf = np.full((n_out, 3), np.nan, dtype=np.float32)
        keypoints_3d_buf = np.full((n_out, 21, 3), np.nan, dtype=np.float32)
        pred_cam_buf = np.full((n_out, 3), np.nan, dtype=np.float32)
        valid_buf = np.zeros(n_out, dtype=bool)
        for rank in range(world_size):
            chunk = np.load(tmp_dir / f"mano_chunk_{rank}.npz")
            cs = int((n_out * rank) // world_size)
            ce = int((n_out * (rank + 1)) // world_size)
            betas_buf[cs:ce] = chunk["betas"]
            global_orient_buf[cs:ce] = chunk["global_orient"]
            hand_pose_buf[cs:ce] = chunk["hand_pose"]
            transl_buf[cs:ce] = chunk["transl"]
            keypoints_3d_buf[cs:ce] = chunk["keypoints_3d"]
            pred_cam_buf[cs:ce] = chunk["pred_cam"]
            valid_buf[cs:ce] = chunk["valid"]

    # ── Save final output ──
    frame_idx_buf = np.arange(n_out, dtype=np.int32)
    ts_buf = np.asarray([_timestamp_at(timestamps_us, idx) for idx in range(n_out)], dtype=np.int64)

    out_dir = session_dir / "wilor_mano"
    out_dir.mkdir(exist_ok=True)
    for name, data in [
        ("mano_betas", betas_buf),
        ("mano_global_orient", global_orient_buf),
        ("mano_hand_pose", hand_pose_buf),
        ("mano_transl", transl_buf),
        ("mano_keypoints_3d", keypoints_3d_buf),
        ("pred_cam", pred_cam_buf),
        ("frame_indices", frame_idx_buf),
        ("timestamps_us", ts_buf),
        ("valid", valid_buf),
        ("bbox_center", centers),
        ("bbox_size", bbox_sizes),
        ("bbox_detected", bbox_detected),
        ("bbox_source", bbox_source),
    ]:
        np.save(out_dir / f"{name}.npy", data)
        print(f"saved {name}.npy {data.shape} {data.dtype}")

    metadata = {
        "pipeline": "wilor_mano_bbox_interp_v2",
        "session_dir": str(session_dir),
        "zed_dir": zed_dir.name,
        "hand": hand,
        "devices": devices,
        "dtype": dtype_str,
        "frame_batch": frame_batch,
        "yolo_conf": yolo_conf,
        "yolo_input_height": yolo_input_height,
        "n_total": int(n_out),
        "n_detected_bbox": n_bbox_detected,
        "n_interpolated_bbox": n_bbox_interpolated,
        "n_edge_filled_bbox": n_bbox_edge_filled,
        "n_no_det": int(total_no_det),
        "n_wrong_hand": int(total_wrong_hand),
        "n_valid_mano": int(valid_buf.sum()),
        "bbox_source_semantics": {
            "0": "detected",
            "1": "interpolated_between_detected_neighbors",
            "2": "edge_filled_from_nearest_detected",
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2)[:2000])
    print(f"\nDone. Output: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="GPU device IDs (default: all visible)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--wilor-pretrained-dir", default=None)
    parser.add_argument("--frame-batch", type=int, default=128)
    parser.add_argument("--yolo-conf", type=float, default=0.1)
    parser.add_argument("--yolo-input-height", type=int, default=512)
    args = parser.parse_args()

    devices = args.devices or list(range(torch.cuda.device_count()))
    if not devices:
        raise SystemExit("No CUDA devices available")

    run_pipeline(
        session_dir=args.session.resolve(),
        hand=args.hand,
        devices=devices,
        dtype_str=args.dtype,
        wilor_pretrained_dir=args.wilor_pretrained_dir,
        frame_batch=args.frame_batch,
        yolo_conf=args.yolo_conf,
        yolo_input_height=args.yolo_input_height,
    )


if __name__ == "__main__":
    main()
