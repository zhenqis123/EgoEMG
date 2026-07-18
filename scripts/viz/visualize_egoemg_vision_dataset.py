"""Visualize samples produced by EgoEmgVisionDataset.

This script is intentionally dataset-centric. Unlike
`scripts/viz/visualize_egoemg_mesh.py`, which validates the world-space MANO mesh
overlay pipeline, this tool inspects the actual WiLoR training samples emitted
by `emg2pose.datasets.egoemg_vision_dataset.EgoEmgVisionDataset`.

For each selected dataset sample it renders a side-by-side debug canvas:

1. The dataset-aligned raw frame with:
   - mirrored left-hand frames exactly as the dataset uses them;
   - projected MANO joints (`orig_keypoints_2d`);
   - mocap markers (`orig_markers_2d`);
   - training bbox.
2. The normalized training patch after denormalization with:
   - patch-space 2D keypoints reconstructed from `keypoints_2d`.

Usage:
    python scripts/viz/visualize_egoemg_vision_dataset.py \
        --memmap-dir data/EgoEMG_memmap \
        --video-root data/EgoEMG \
        --output-dir /tmp/egoemg_vision_dataset_viz \
        --num-samples 16 \
        --target-hand both
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# Headless servers often have unwritable default matplotlib/fontconfig cache
# locations. Importing WiLoR transitively can trigger those caches, so point
# them at a writable temp directory before importing the dataset module.
_runtime_cache_root = Path(tempfile.gettempdir()) / "emg2pose_viz_runtime"
_runtime_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache_root / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache_root / "xdg-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from emg2pose.datasets.egoemg_vision_dataset import EgoEmgVisionDataset


JOINT_COLOR_BGR = (0, 220, 0)
MARKER_COLOR_BGR = (0, 255, 255)
BBOX_COLOR_BGR = (255, 180, 0)
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_SHADOW_BGR = (20, 20, 20)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-dir", default="data/EgoEMG_memmap")
    parser.add_argument("--video-root", default="data/EgoEMG")
    parser.add_argument("--allintra-root", default=None)
    parser.add_argument("--allintra-suffix", default="_allintra")
    parser.add_argument(
        "--vision-index-dir",
        default=None,
        help="Defaults to <memmap-dir>/vision_index.",
    )
    parser.add_argument(
        "--auto-build-index",
        action="store_true",
        help="Build the sidecar vision index if it is missing.",
    )
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="Optional override for EgoEMG webcam calibration json.",
    )
    parser.add_argument("--output-dir", default="/tmp/egoemg_vision_dataset_viz")
    parser.add_argument("--target-hand", default="both", choices=["left", "right", "both"])
    parser.add_argument(
        "--allowed-episode-ids",
        nargs="*",
        default=None,
        help="Optional list of episode_id strings to keep.",
    )
    parser.add_argument(
        "--allowed-subjects",
        nargs="*",
        default=None,
        help="Optional list of subject names to keep.",
    )
    parser.add_argument(
        "--allowed-splits",
        nargs="*",
        default=None,
        help="Optional list of split names to keep.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit dataset indices. Overrides start-index/num-samples.",
    )
    parser.add_argument("--joint-radius", type=int, default=3)
    parser.add_argument("--marker-radius", type=int, default=3)
    parser.add_argument("--bbox-line-width", type=int, default=2)
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only render the dataset-aligned raw frame panel.",
    )
    parser.add_argument(
        "--max-panel-width",
        type=int,
        default=1280,
        help="Resize the final side-by-side canvas if wider than this.",
    )
    return parser.parse_args()


def _draw_points(
    image_bgr: np.ndarray,
    points_xyc: np.ndarray,
    color_bgr: tuple[int, int, int],
    radius: int,
) -> np.ndarray:
    out = image_bgr.copy()
    for point in points_xyc:
        if point.shape[0] < 3 or point[2] <= 0:
            continue
        if not np.isfinite(point[:2]).all():
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(out, (x, y), radius, color_bgr, -1, lineType=cv2.LINE_AA)
    return out


def _draw_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    color_bgr: tuple[int, int, int],
    line_width: int,
) -> np.ndarray:
    out = image_bgr.copy()
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox_xyxy]
    cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr, line_width, lineType=cv2.LINE_AA)
    return out


def _draw_text_block(
    image_bgr: np.ndarray,
    lines: list[str],
    origin_xy: tuple[int, int] = (20, 30),
    line_height: int = 28,
) -> np.ndarray:
    out = image_bgr.copy()
    x, y = origin_xy
    for line in lines:
        cv2.putText(
            out,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_SHADOW_BGR,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return out


def _denormalize_patch_rgb(img_chw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    img = img_chw.astype(np.float32).copy()
    for channel_idx in range(3):
        img[channel_idx] = img[channel_idx] * float(std[channel_idx]) + float(mean[channel_idx])
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def _patch_keypoints_to_pixels(keypoints_2d: np.ndarray, patch_size: int) -> np.ndarray:
    out = keypoints_2d.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] + 0.5) * float(patch_size)
    out[:, 1] = (out[:, 1] + 0.5) * float(patch_size)
    return out


def _build_raw_frame_panel(
    dataset: EgoEmgVisionDataset,
    sample: dict,
    joint_radius: int,
    marker_radius: int,
    bbox_line_width: int,
) -> np.ndarray:
    frame_bgr = np.asarray(sample["frame_bgr"], dtype=np.uint8)
    is_mirrored = float(sample["raw_right"]) == 0.0
    video_w = frame_bgr.shape[1]

    if is_mirrored:
        frame_bgr = np.ascontiguousarray(frame_bgr[:, ::-1])

    def _unmirror_xy(pts: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        if is_mirrored:
            pts[:, 0] = (video_w - 1) - pts[:, 0]
        return pts

    panel = frame_bgr
    bbox = np.asarray(sample["bbox"], dtype=np.float32)
    if is_mirrored:
        x0, y0, x1, y1 = bbox
        bbox = np.array([(video_w - 1) - x1, y0, (video_w - 1) - x0, y1], dtype=np.float32)
    panel = _draw_bbox(panel, bbox, BBOX_COLOR_BGR, bbox_line_width)
    panel = _draw_points(
        panel,
        _unmirror_xy(np.asarray(sample["orig_markers_2d"], dtype=np.float32)),
        MARKER_COLOR_BGR,
        marker_radius,
    )
    panel = _draw_points(
        panel,
        _unmirror_xy(np.asarray(sample["orig_keypoints_2d"], dtype=np.float32)),
        JOINT_COLOR_BGR,
        joint_radius,
    )

    joints_valid = int((np.asarray(sample["orig_keypoints_2d"])[:, 2] > 0).sum())
    markers_valid = int((np.asarray(sample["orig_markers_2d"])[:, 2] > 0).sum())
    lines = [
        f"dataset_idx={int(sample['_dataset_index'])} hand={sample['target_hand']}",
        f"episode={sample['episode_id']} subject={sample['episode_subject']}",
        f"frame_idx={int(sample['frame_index'])} video_frame={int(sample['video_frame_index'])}",
        f"bbox_source={sample['bbox_source_name']} joints={joints_valid}/21 markers={markers_valid}/21",
        f"raw_right={float(sample['raw_right']):.0f} canonical_right={float(sample['is_right']):.0f}",
    ]
    return _draw_text_block(panel, lines)


def _build_patch_panel(
    dataset: EgoEmgVisionDataset,
    sample: dict,
    joint_radius: int,
) -> np.ndarray:
    patch_rgb = _denormalize_patch_rgb(
        np.asarray(sample["img"], dtype=np.float32),
        dataset.mean,
        dataset.std,
    )
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    patch_points = _patch_keypoints_to_pixels(
        np.asarray(sample["keypoints_2d"], dtype=np.float32),
        int(dataset.patch_size),
    )
    panel = _draw_points(patch_bgr, patch_points, JOINT_COLOR_BGR, joint_radius)
    patch_valid = int((patch_points[:, 2] > 0).sum())
    lines = [
        f"patch_size={int(dataset.patch_size)}",
        f"keypoints_2d in patch={patch_valid}/21",
        "coords reconstructed from normalized dataset output",
    ]
    return _draw_text_block(panel, lines)


def _make_canvas(raw_panel: np.ndarray, patch_panel: np.ndarray, max_panel_width: int) -> np.ndarray:
    raw_h = raw_panel.shape[0]
    scale = raw_h / float(patch_panel.shape[0])
    patch_w = max(1, int(round(patch_panel.shape[1] * scale)))
    patch_h = raw_h
    patch_resized = cv2.resize(patch_panel, (patch_w, patch_h), interpolation=cv2.INTER_NEAREST)
    spacer = np.full((raw_h, 24, 3), 24, dtype=np.uint8)
    canvas = np.concatenate([raw_panel, spacer, patch_resized], axis=1)

    if canvas.shape[1] <= max_panel_width:
        return canvas
    scale_out = max_panel_width / float(canvas.shape[1])
    out_w = max(1, int(round(canvas.shape[1] * scale_out)))
    out_h = max(1, int(round(canvas.shape[0] * scale_out)))
    return cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _resolve_indices(dataset_len: int, start_index: int, num_samples: int, sample_indices: list[int] | None) -> list[int]:
    if sample_indices:
        indices = sample_indices
    else:
        indices = list(range(start_index, min(start_index + num_samples, dataset_len)))
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Dataset index out of range: {idx} not in [0, {dataset_len})")
    return indices


def _log(message: str) -> None:
    print(f"[egoemg_vision_viz] {message}", flush=True)


def _requested_index_limit(
    start_index: int,
    num_samples: int,
    sample_indices: list[int] | None,
) -> int:
    if sample_indices:
        return max(sample_indices) + 1
    return start_index + num_samples


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output dir: {output_dir}")
    _log(
        "Config: "
        f"memmap_dir={Path(args.memmap_dir).resolve()} "
        f"video_root={Path(args.video_root).resolve()} "
        f"target_hand={args.target_hand} "
        f"raw_only={args.raw_only}"
    )
    requested_limit = _requested_index_limit(
        args.start_index,
        args.num_samples,
        args.sample_indices,
    )
    _log(f"Index build limit: first {requested_limit} matching samples")

    _log("Building EgoEmgVisionDataset")
    t_dataset = time.perf_counter()
    dataset = EgoEmgVisionDataset(
        memmap_dir=Path(args.memmap_dir),
        video_root=Path(args.video_root),
        allintra_root=Path(args.allintra_root) if args.allintra_root else None,
        allintra_suffix=args.allintra_suffix,
        vision_index_dir=Path(args.vision_index_dir) if args.vision_index_dir else None,
        auto_build_index=args.auto_build_index,
        calibration_path=Path(args.calibration_path) if args.calibration_path else None,
        allowed_episode_ids=args.allowed_episode_ids,
        allowed_subjects=args.allowed_subjects,
        allowed_splits=args.allowed_splits,
        target_hand=args.target_hand,
        stride=args.stride,
        index_limit=requested_limit,
        patch_size=args.patch_size,
        do_augment=False,
        return_frame_bgr=True,
        log_init_timing=True,
    )
    _log(
        f"Dataset ready: len={len(dataset)} "
        f"vision_index_dir={dataset.vision_index_dir.resolve()} "
        f"allintra_root={dataset.allintra_root.resolve()} "
        f"elapsed_s={time.perf_counter() - t_dataset:.3f}"
    )

    indices = _resolve_indices(len(dataset), args.start_index, args.num_samples, args.sample_indices)
    _log(f"Rendering {len(indices)} samples")

    for render_i, dataset_idx in enumerate(indices, start=1):
        _log(f"[{render_i}/{len(indices)}] Loading dataset_idx={dataset_idx}")
        t_sample = time.perf_counter()
        sample = dataset[dataset_idx]
        sample["_dataset_index"] = np.int64(dataset_idx)
        _log(
            f"[{render_i}/{len(indices)}] Loaded "
            f"episode={sample['episode_id']} frame={int(sample['frame_index'])} "
            f"video_frame={int(sample['video_frame_index'])} hand={sample['target_hand']} "
            f"elapsed_s={time.perf_counter() - t_sample:.3f}"
        )
        raw_panel = _build_raw_frame_panel(
            dataset,
            sample,
            joint_radius=args.joint_radius,
            marker_radius=args.marker_radius,
            bbox_line_width=args.bbox_line_width,
        )
        if args.raw_only:
            canvas = raw_panel
        else:
            patch_panel = _build_patch_panel(
                dataset,
                sample,
                joint_radius=args.joint_radius,
            )
            canvas = _make_canvas(raw_panel, patch_panel, max_panel_width=args.max_panel_width)

        episode_id = str(sample["episode_id"])
        frame_idx = int(sample["frame_index"])
        hand = str(sample["target_hand"])
        out_path = output_dir / (
            f"sample_{dataset_idx:06d}_ep_{episode_id}_frame_{frame_idx:08d}_{hand}.png"
        )
        cv2.imwrite(str(out_path), canvas)
        _log(f"[{render_i}/{len(indices)}] Wrote {out_path}")

    _log("Done")


if __name__ == "__main__":
    main()
