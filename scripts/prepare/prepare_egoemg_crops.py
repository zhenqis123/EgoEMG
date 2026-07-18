"""Per-episode crop preparation for EgoEMG vision data.

For each episode video, decode frames, project mocap keypoints, crop hand
patches, and store as JPEG in a per-episode LMDB. No split/stride logic —
that is handled at training time by the dataset.

Output structure:
    {output_dir}/
        manifest.json
        episode_000000.lmdb/
        episode_000001.lmdb/
        ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import lmdb
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

WILOR_PATH = Path(__file__).resolve().parents[1] / "WiLoR"
if str(WILOR_PATH) not in sys.path:
    sys.path.insert(0, str(WILOR_PATH))

from decord import VideoReader, gpu

from emg2pose.datasets.egoemg_vision_dataset import (
    _build_intrinsics_and_frame_mapper,
    _expand_to_aspect_ratio,
    _generate_image_patch_cv2,
    _get_bbox,
    _load_memmap,
    _map_processed_points_to_raw,
    _project_world_points,
)
from emg2pose.video_io import resolve_allintra_video_path

DECODE_BATCH = 4096
NUM_CROP_WORKERS = 16


def _decode_bytes(arr: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


def _progress(iterable, *, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _load_calibration(calibration_path: Path):
    with calibration_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    return (
        np.asarray(calib["camera_matrix"], dtype=np.float64),
        np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1),
        int(calib["image_width"]),
        int(calib["image_height"]),
    )


def _compute_intrinsics(frame_bgr, K_calib, dist_calib, calib_w, calib_h):
    video_h, video_w = frame_bgr.shape[:2]
    return _build_intrinsics_and_frame_mapper(
        K_calib, dist_calib, calib_w, calib_h, video_w, video_h, frame_bgr,
    )


def _check_hand_valid(
    frame_memmaps, frame_idx, hand_idx, hand_name,
    K, dist, intrinsics_info, video_w, video_h,
):
    """Return True if this hand has >=2 valid projected keypoints."""
    if not bool(frame_memmaps["generated_label_valid"][frame_idx, hand_idx]):
        return False
    T_W_C = np.eye(4, dtype=np.float64)
    t12 = np.asarray(frame_memmaps["mocap_webcam_transform"][frame_idx], dtype=np.float64)
    T_W_C[:3, :3] = t12[:9].reshape(3, 3)
    T_W_C[:3, 3] = t12[9:12]
    marker_world = np.asarray(
        frame_memmaps[f"mocap_{hand_name}_keypoints"][frame_idx], dtype=np.float64,
    )
    marker_valid = np.asarray(
        frame_memmaps[f"mocap_{hand_name}_valid"][frame_idx], dtype=bool,
    )
    marker_proc, marker_depth_valid = _project_world_points(marker_world, T_W_C, K, dist)
    marker_raw = _map_processed_points_to_raw(marker_proc, intrinsics_info)
    marker_in_image = (
        (marker_raw[:, 0] >= 0) & (marker_raw[:, 0] < video_w)
        & (marker_raw[:, 1] >= 0) & (marker_raw[:, 1] < video_h)
    )
    n_valid = int((marker_valid & marker_depth_valid & marker_in_image).sum())
    return n_valid >= 2


def _crop_hand(
    frame_bgr, frame_memmaps, frame_idx, hand_idx, hand_name,
    K, dist, intrinsics_info, patch_size,
):
    """Crop a hand patch from the frame. Returns BGR image patch."""
    video_h, video_w = frame_bgr.shape[:2]
    T_W_C = np.eye(4, dtype=np.float64)
    t12 = np.asarray(frame_memmaps["mocap_webcam_transform"][frame_idx], dtype=np.float64)
    T_W_C[:3, :3] = t12[:9].reshape(3, 3)
    T_W_C[:3, 3] = t12[9:12]
    marker_world = np.asarray(
        frame_memmaps[f"mocap_{hand_name}_keypoints"][frame_idx], dtype=np.float64,
    )
    marker_valid = np.asarray(
        frame_memmaps[f"mocap_{hand_name}_valid"][frame_idx], dtype=bool,
    )
    label_valid = bool(frame_memmaps["generated_label_valid"][frame_idx, hand_idx])
    is_left = hand_idx == 0

    marker_proc, marker_depth_valid = _project_world_points(marker_world, T_W_C, K, dist)
    marker_raw = _map_processed_points_to_raw(marker_proc, intrinsics_info)
    marker_in_image = (
        (marker_raw[:, 0] >= 0) & (marker_raw[:, 0] < video_w)
        & (marker_raw[:, 1] >= 0) & (marker_raw[:, 1] < video_h)
    )
    markers_2d_xy = marker_raw.astype(np.float32)
    if is_left:
        frame_bgr = np.ascontiguousarray(frame_bgr[:, ::-1])
        markers_2d_xy[:, 0] = (video_w - 1) - markers_2d_xy[:, 0]

    markers_conf = (marker_valid & marker_depth_valid & marker_in_image).astype(np.float32)
    if not label_valid:
        markers_conf[:] = 0.0
    keypoints_2d = np.concatenate([markers_2d_xy, markers_conf[:, None]], axis=-1)

    valid_for_bbox = keypoints_2d[:, 2] > 0
    if valid_for_bbox.sum() < 2:
        center = np.array([video_w / 2.0, video_h / 2.0], dtype=np.float32)
        scale = np.array([video_w / 3.0, video_h / 3.0], dtype=np.float32)
    else:
        center, scale = _get_bbox(keypoints_2d, rescale=1.2)
    scale = _expand_to_aspect_ratio(scale, (192, 256))
    bbox_size = float(max(scale[0], scale[1]))
    img_patch_bgr, _ = _generate_image_patch_cv2(
        frame_bgr, float(center[0]), float(center[1]),
        bbox_size, bbox_size, patch_size, patch_size,
        do_flip=False, scale=1.0, rot=0.0,
    )
    return img_patch_bgr


def _process_episode(
    ep_idx: int,
    ep_id: str,
    video_path: str,
    start_idx: int,
    end_idx: int,
    frame_memmaps: dict,
    K: np.ndarray,
    dist: np.ndarray,
    intrinsics_info: dict,
    video_w: int,
    video_h: int,
    output_dir: Path,
    patch_size: int,
    jpeg_quality: int,
    gpu_id: int,
    progress: bool,
):
    lmdb_path = output_dir / f"{ep_id}.lmdb"
    done_path = output_dir / f"{ep_id}.done"

    vfi_memmap = frame_memmaps["image_webcam_frame_index"]
    ep_vfi = np.asarray(vfi_memmap[start_idx:end_idx], dtype=np.int32)

    vfi_to_frame: dict[int, int] = {}
    for offset, vfi in enumerate(ep_vfi):
        vfi = int(vfi)
        if vfi >= 0 and vfi not in vfi_to_frame:
            vfi_to_frame[vfi] = start_idx + offset

    unique_vfis = sorted(vfi_to_frame.keys())
    n_video_frames = len(unique_vfis)
    print(f"  [{ep_id}] {n_video_frames} unique video frames, "
          f"memmap range [{start_idx}, {end_idx})", flush=True)

    valid_work: list[tuple[int, int, int, str]] = []
    for vfi in unique_vfis:
        fi = vfi_to_frame[vfi]
        for hand_idx, hand_name in [(0, "left"), (1, "right")]:
            if _check_hand_valid(
                frame_memmaps, fi, hand_idx, hand_name,
                K, dist, intrinsics_info, video_w, video_h,
            ):
                valid_work.append((vfi, fi, hand_idx, hand_name))

    print(f"  [{ep_id}] {len(valid_work)} valid crops to produce "
          f"(of {n_video_frames * 2} possible)", flush=True)

    if not valid_work:
        env = lmdb.open(str(lmdb_path), map_size=1024 * 1024)
        env.close()
        done_path.write_text(json.dumps({"num_crops": 0, "num_video_frames": n_video_frames}))
        return 0

    vr = VideoReader(str(video_path), ctx=gpu(gpu_id))
    n_actual_frames = len(vr)

    vfi_to_work: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for vfi, fi, hand_idx, hand_name in valid_work:
        if vfi < n_actual_frames:
            vfi_to_work[vfi].append((fi, hand_idx, hand_name))

    decode_vfis = sorted(vfi_to_work.keys())
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    map_size = max(len(valid_work) * 20_000, 10 * 1024 * 1024)
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    txn = env.begin(write=True)
    total_written = 0

    pbar = _progress(
        range(len(decode_vfis)), enabled=progress,
        desc=f"{ep_id}", unit="vf", total=len(decode_vfis),
    )
    pbar_iter = iter(pbar)

    for batch_start in range(0, len(decode_vfis), DECODE_BATCH):
        batch_vfis = decode_vfis[batch_start:batch_start + DECODE_BATCH]
        batch_frames = vr.get_batch(batch_vfis).asnumpy()

        work_items = []
        for local_i, vfi in enumerate(batch_vfis):
            frame_bgr = cv2.cvtColor(batch_frames[local_i], cv2.COLOR_RGB2BGR)
            for fi, hand_idx, hand_name in vfi_to_work[vfi]:
                key = f"{vfi:08d}_{'L' if hand_idx == 0 else 'R'}"
                work_items.append((key, frame_bgr, fi, hand_idx, hand_name))

        def _encode_one(item):
            key, frame_bgr, fi, hand_idx, hand_name = item
            patch = _crop_hand(
                frame_bgr, frame_memmaps, fi, hand_idx, hand_name,
                K, dist, intrinsics_info, patch_size,
            )
            ok, encoded = cv2.imencode(".jpg", patch, jpeg_params)
            if not ok:
                raise RuntimeError(f"JPEG encode failed for {key}")
            return key, encoded.tobytes()

        with ThreadPoolExecutor(max_workers=NUM_CROP_WORKERS) as pool:
            futures = [pool.submit(_encode_one, item) for item in work_items]
            for fut in as_completed(futures):
                key, jpeg_bytes = fut.result()
                txn.put(key.encode("ascii"), jpeg_bytes)
                total_written += 1

        if total_written % 5000 < len(work_items):
            txn.commit()
            txn = env.begin(write=True)

        for _ in batch_vfis:
            try:
                next(pbar_iter)
            except StopIteration:
                break

    txn.commit()
    env.sync()
    env.close()

    done_path.write_text(json.dumps({
        "num_crops": total_written,
        "num_video_frames": n_video_frames,
        "num_actual_video_frames": n_actual_frames,
    }))
    print(f"  [{ep_id}] done: {total_written} crops written", flush=True)
    return total_written


def main():
    parser = argparse.ArgumentParser(description="Per-episode crop preparation.")
    parser.add_argument("--memmap-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--allintra-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--episode-ids", type=str, default=None,
                        help="Comma-separated episode indices (e.g. 0,1,2)")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()

    if args.calibration_path is None:
        args.calibration_path = (
            args.video_root / "reprojection_assets" / "GX010023_standard_calibration.json"
        )

    t0 = time.perf_counter()
    print(f"Loading metadata from {args.memmap_dir} ...", flush=True)
    with (args.memmap_dir / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    metadata = np.load(args.memmap_dir / "metadata.npz", allow_pickle=False)
    ep_ids = _decode_bytes(metadata["episode_id"])
    ep_video_paths = _decode_bytes(metadata["episode_webcam_video_path"])
    ep_starts = metadata["episode_start_idx"].astype(np.int64)
    ep_ends = metadata["episode_end_idx"].astype(np.int64)
    num_episodes = len(ep_ids)
    print(f"  {num_episodes} episodes, {time.perf_counter()-t0:.1f}s", flush=True)

    print("Loading memmaps ...", flush=True)
    fields = [
        "image_webcam_frame_index", "mocap_webcam_transform",
        "generated_label_valid",
        "mocap_left_keypoints", "mocap_right_keypoints",
        "mocap_left_valid", "mocap_right_valid",
    ]
    frame_memmaps = {
        f: _load_memmap(args.memmap_dir, manifest["fields"][f]) for f in fields
    }
    print(f"  memmaps loaded, {time.perf_counter()-t0:.1f}s", flush=True)

    print("Loading calibration ...", flush=True)
    K_calib, dist_calib, calib_w, calib_h = _load_calibration(args.calibration_path)
    print(f"  calibration loaded, {time.perf_counter()-t0:.1f}s", flush=True)

    if args.episode_ids is not None:
        episode_indices = [int(x) for x in args.episode_ids.split(",")]
    else:
        episode_indices = list(range(num_episodes))
    if args.num_workers > 1:
        episode_indices = [i for i in episode_indices if i % args.num_workers == args.worker_id]

    print(f"Will process {len(episode_indices)} episodes "
          f"(worker {args.worker_id}/{args.num_workers})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    K, dist, intrinsics_info, video_w, video_h = None, None, None, None, None
    manifest_path = args.output_dir / "manifest.json"

    total_crops = 0
    for ep_idx in episode_indices:
        ep_id = ep_ids[ep_idx]
        done_path = args.output_dir / f"{ep_id}.done"
        if done_path.exists() and not args.overwrite:
            print(f"[{ep_id}] already done, skipping", flush=True)
            continue

        if args.overwrite:
            lmdb_path = args.output_dir / f"{ep_id}.lmdb"
            if lmdb_path.exists():
                import shutil
                shutil.rmtree(lmdb_path)
            if done_path.exists():
                done_path.unlink()

        raw_video_path = ep_video_paths[ep_idx]
        video_path = resolve_allintra_video_path(
            raw_video_path=raw_video_path,
            data_root=args.video_root,
            allintra_root=args.allintra_root,
        )

        if K is None:
            vr_tmp = VideoReader(str(video_path), ctx=gpu(args.gpu_id))
            first_frame = cv2.cvtColor(vr_tmp[0].asnumpy(), cv2.COLOR_RGB2BGR)
            K, dist, intrinsics_info = _compute_intrinsics(
                first_frame, K_calib, dist_calib, calib_w, calib_h,
            )
            video_h, video_w = first_frame.shape[:2]
            del vr_tmp
            global_manifest = {
                "version": 2,
                "format": "per_episode_crops",
                "patch_size": args.patch_size,
                "jpeg_quality": args.jpeg_quality,
                "intrinsics": {
                    "K": K.tolist(),
                    "dist": dist.tolist(),
                    "info": intrinsics_info,
                    "video_w": video_w,
                    "video_h": video_h,
                },
                "num_episodes": num_episodes,
                "episode_ids": ep_ids,
            }
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(global_manifest, f, indent=2)

        print(f"[{ep_id}] processing ...", flush=True)
        n = _process_episode(
            ep_idx=ep_idx,
            ep_id=ep_id,
            video_path=str(video_path),
            start_idx=int(ep_starts[ep_idx]),
            end_idx=int(ep_ends[ep_idx]),
            frame_memmaps=frame_memmaps,
            K=K, dist=dist,
            intrinsics_info=intrinsics_info,
            video_w=video_w, video_h=video_h,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            jpeg_quality=args.jpeg_quality,
            gpu_id=args.gpu_id,
            progress=args.progress,
        )
        total_crops += n

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {total_crops} total crops in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
