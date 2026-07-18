#!/usr/bin/env python3
"""Summarize and visualize frames missing saved WiLoR MANO labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def _find_runs(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indices.size == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty
    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[indices[0], indices[breaks + 1]].astype(np.int64)
    ends = np.r_[indices[breaks], indices[-1]].astype(np.int64)
    lengths = ends - starts + 1
    return starts, ends, lengths


def _select_representatives(
    starts: np.ndarray,
    ends: np.ndarray,
    lengths: np.ndarray,
    invalid: np.ndarray,
    max_frames: int,
) -> list[int]:
    selected: list[int] = []

    # Cover the longest gaps with start/mid/end points.
    for run_idx in np.argsort(lengths)[::-1]:
        start = int(starts[run_idx])
        end = int(ends[run_idx])
        mid = (start + end) // 2
        for frame_idx in (start, mid, end):
            if frame_idx not in selected:
                selected.append(frame_idx)
            if len(selected) >= max_frames:
                return sorted(selected)

    # Add evenly spaced invalid frames if room remains.
    if len(selected) < max_frames and invalid.size:
        n_extra = max_frames - len(selected)
        spaced = invalid[np.linspace(0, invalid.size - 1, n_extra, dtype=np.int64)]
        for frame_idx in spaced.tolist():
            if int(frame_idx) not in selected:
                selected.append(int(frame_idx))
    return sorted(selected[:max_frames])


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
        raise RuntimeError(f"Sequential read missed requested frames: {missing}")
    return frames


def _run_yolo(
    frame_bgr: np.ndarray,
    detector: YOLO,
    conf: float,
    input_height: int,
) -> list[dict]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_h, image_w = frame_rgb.shape[:2]
    if input_height > 0:
        yolo_h = input_height
        yolo_w = max(int(image_w * yolo_h / image_h), 1)
        scale_x = image_w / float(yolo_w)
        scale_y = image_h / float(yolo_h)
        frame_yolo = cv2.resize(frame_rgb, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale_x = 1.0
        scale_y = 1.0
        frame_yolo = frame_rgb
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


def _classify(detections: list[dict], target_hand: str) -> str:
    if not detections:
        return "no_detection"
    if any(det["hand"] == target_hand for det in detections):
        return "target_detected_now"
    return "wrong_hand_only"


def _draw(frame_bgr: np.ndarray, detections: list[dict], target_hand: str, title: str) -> np.ndarray:
    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        color = (0, 255, 0) if det["hand"] == target_hand else (255, 0, 0)
        label = f"{det['hand']} {det['score']:.2f}"
        if det["hand"] == target_hand:
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
            label,
            (int(round(x1)), max(24, int(round(y1)) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _make_contact_sheet(image_paths: list[Path], output_path: Path, thumb_width: int, cols: int) -> None:
    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(
            image,
            (thumb_width, int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        thumbs.append(thumb)
    if not thumbs:
        return
    rows = []
    for start in range(0, len(thumbs), cols):
        row_imgs = thumbs[start : start + cols]
        max_h = max(img.shape[0] for img in row_imgs)
        padded = []
        for img in row_imgs:
            if img.shape[0] < max_h:
                pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
                img = np.vstack([img, pad])
            padded.append(img)
        while len(padded) < cols:
            padded.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(padded))
    cv2.imwrite(str(output_path), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-hand", default="right", choices=["right", "left"])
    parser.add_argument("--detector-path", type=Path, default=Path("data/pretrained_models/detector.pt"))
    parser.add_argument("--yolo-conf", type=float, default=0.3)
    parser.add_argument(
        "--yolo-input-height",
        type=int,
        default=256,
        help="YOLO input height. Use 0 to run detector at original frame resolution.",
    )
    parser.add_argument("--max-viz", type=int, default=96)
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    session = args.session
    wilor_dir = session / "wilor_mano"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    valid = np.load(wilor_dir / "valid.npy").astype(bool)
    frame_indices = np.load(wilor_dir / "frame_indices.npy").astype(np.int64)
    timestamps_us = np.load(wilor_dir / "timestamps_us.npy").astype(np.int64)
    invalid_out_idx = np.where(~valid)[0].astype(np.int64)
    invalid_frames = frame_indices[invalid_out_idx]
    starts, ends, lengths = _find_runs(invalid_frames)

    with open(output / "missing_frames.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["out_idx", "frame_idx", "timestamp_us"])
        for out_idx in invalid_out_idx.tolist():
            writer.writerow([out_idx, int(frame_indices[out_idx]), int(timestamps_us[out_idx])])

    with open(output / "missing_runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "start_frame", "end_frame", "length", "start_timestamp_us", "end_timestamp_us"])
        for run_id, (start, end, length) in enumerate(zip(starts, ends, lengths)):
            writer.writerow(
                [
                    run_id,
                    int(start),
                    int(end),
                    int(length),
                    int(timestamps_us[int(start)]),
                    int(timestamps_us[int(end)]),
                ]
            )

    selected_frames = _select_representatives(starts, ends, lengths, invalid_frames, args.max_viz)
    zed_dirs = sorted(path for path in session.iterdir() if path.is_dir() and path.name.startswith("ZED_"))
    if not zed_dirs:
        raise FileNotFoundError(f"No ZED_* directory under {session}")
    video_path = zed_dirs[0] / "rgb.mkv"
    frames = _read_frames_sequential(video_path, selected_frames)

    detector = YOLO(str(args.detector_path))
    visualized: list[dict] = []
    image_paths: list[Path] = []
    for frame_idx in selected_frames:
        frame = frames[frame_idx]
        detections = _run_yolo(frame, detector, args.yolo_conf, args.yolo_input_height)
        reason = _classify(detections, args.target_hand)
        out_idx = int(np.where(frame_indices == frame_idx)[0][0])
        title = f"frame={frame_idx} out={out_idx} {reason}"
        image = _draw(frame, detections, args.target_hand, title)
        image_path = output / f"missing_frame_{frame_idx:06d}_out_{out_idx:06d}_{reason}.png"
        cv2.imwrite(str(image_path), image)
        image_paths.append(image_path)
        visualized.append(
            {
                "frame_idx": int(frame_idx),
                "out_idx": out_idx,
                "timestamp_us": int(timestamps_us[out_idx]),
                "reason_now": reason,
                "detections": detections,
                "png": str(image_path),
            }
        )

    _make_contact_sheet(image_paths, output / "missing_frames_contact_sheet.png", args.thumb_width, args.cols)
    reason_counts: dict[str, int] = {}
    for item in visualized:
        reason_counts[item["reason_now"]] = reason_counts.get(item["reason_now"], 0) + 1

    summary = {
        "session": str(session),
        "video_path": str(video_path),
        "total_frames": int(len(valid)),
        "valid_frames": int(valid.sum()),
        "missing_frames": int((~valid).sum()),
        "missing_fraction": float((~valid).mean()),
        "missing_runs": int(len(starts)),
        "max_run_length": int(lengths.max()) if lengths.size else 0,
        "median_run_length": float(np.median(lengths)) if lengths.size else 0.0,
        "mean_run_length": float(lengths.mean()) if lengths.size else 0.0,
        "longest_runs": [
            {
                "start_frame": int(starts[idx]),
                "end_frame": int(ends[idx]),
                "length": int(lengths[idx]),
            }
            for idx in np.argsort(lengths)[::-1][:20].tolist()
        ],
        "visualized_frames": visualized,
        "visualized_reason_counts": reason_counts,
        "yolo_conf": args.yolo_conf,
        "yolo_input_height": args.yolo_input_height,
        "missing_frames_csv": str(output / "missing_frames.csv"),
        "missing_runs_csv": str(output / "missing_runs.csv"),
        "contact_sheet": str(output / "missing_frames_contact_sheet.png"),
    }
    with open(output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:4000])
    print(f"output_dir={output}")


if __name__ == "__main__":
    main()
