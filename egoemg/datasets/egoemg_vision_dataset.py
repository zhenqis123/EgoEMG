"""EgoEMG -> WiLoR dataset adapter.

This dataset exposes EgoEMG samples in the same supervision format expected by
WiLoR: cropped/normalized images, 2D/3D keypoints, and MANO parameters.

Important semantic note for EgoEMG:
- both hands store MANO poses in MANO-right canonical semantics;
- left-hand world transforms are defined on x-mirrored raw MANO geometry.

To fine-tune WiLoR consistently, every sample is converted to a right-hand
canonical training representation. For left hands this means:
- mirror the full frame horizontally;
- mirror projected 2D keypoints and camera-space 3D keypoints along x;
- keep local MANO articulation in MANO-right semantics;
- convert the local-to-camera global orientation into the mirrored camera frame.
"""

from __future__ import annotations

import json
import os
import random
import time
from heapq import merge as heapq_merge
from itertools import islice
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from egoemg.video_io import (
    DEFAULT_ALLINTRA_SUFFIX,
    DecordVideoReaderCache,
    FfmpegNvdecReaderCache,
    resolve_allintra_video_path,
)


MIRROR_X_3 = np.diag(np.array([-1.0, 1.0, 1.0], dtype=np.float64))
IDENTITY_KP_PERMUTATION = list(range(21))
VISION_INDEX_VERSION = 1

MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "mano": (
        "generated_mano_left_pose",
        "generated_mano_right_pose",
        "generated_label_valid",
    ),
    "joint_angles": (
        "generated_joint_angles_left",
        "generated_joint_angles_right",
    ),
    "mocap_hands": (
        "mocap_left_keypoints",
        "mocap_right_keypoints",
        "mocap_left_valid",
        "mocap_right_valid",
        "mocap_left_wrist_angles_valid",
        "mocap_right_wrist_angles_valid",
        "mocap_left_wrist_pitch",
        "mocap_left_wrist_yaw",
        "mocap_right_wrist_pitch",
        "mocap_right_wrist_yaw",
    ),
    "camera": (
        "mocap_head_transform",
        "mocap_head_tracked",
        "mocap_mano_left_world_transform",
        "mocap_mano_right_world_transform",
    ),
    "video_index": (
        "image_head_frame_index",
        "image_head_stale",
    ),
    "split": ("frame_split_id",),
}


def _decode_bytes(values: np.ndarray) -> list[str]:
    return [
        v.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(v, (bytes, np.bytes_))
        else str(v)
        for v in values
    ]


def _resolve_str_filter(
    values: Sequence[str],
    allowed: Sequence[str] | None,
    label: str,
) -> set[int]:
    if not allowed:
        return set()
    value_to_id = {v: i for i, v in enumerate(values)}
    ids: set[int] = set()
    for a in allowed:
        if a not in value_to_id:
            raise ValueError(f"Unknown {label}: {a}")
        ids.add(int(value_to_id[a]))
    return ids


def _load_memmap(root: Path, info: dict[str, Any]) -> np.memmap:
    return np.memmap(
        str(root / info["filename"]),
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def _sanitize_index_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _default_vision_index_dir(memmap_dir: Path) -> Path:
    return memmap_dir / "vision_index"


def build_egoemg_vision_index(
    *,
    memmap_dir: Path,
    output_dir: Path | None = None,
    chunk_size: int = 1_000_000,
    overwrite: bool = False,
    log_every_episode: bool = True,
) -> Path:
    """Build reusable frame/hand index shards for EgoEmgVisionDataset.

    The generated index stores only valid frame ids grouped by split, hand, and
    episode. Dataset construction can then apply stride and episode/subject
    filters without scanning frame-level memmaps.
    """

    memmap_dir = Path(memmap_dir)
    output_dir = Path(output_dir) if output_dir is not None else _default_vision_index_dir(memmap_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return output_dir
    if overwrite:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    with (memmap_dir / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    metadata = np.load(memmap_dir / "metadata.npz", allow_pickle=False)
    episode_id = _decode_bytes(metadata["episode_id"])
    episode_start_idx = np.asarray(metadata["episode_start_idx"], dtype=np.int64)
    episode_end_idx = np.asarray(metadata["episode_end_idx"], dtype=np.int64)
    split_names = _decode_bytes(metadata["splits_split"])

    frame_split_mm = _load_memmap(memmap_dir, manifest["fields"]["frame_split_id"])
    label_valid_mm = _load_memmap(memmap_dir, manifest["fields"]["generated_label_valid"])
    tracked_mm = _load_memmap(memmap_dir, manifest["fields"]["mocap_head_tracked"])
    stale_mm = _load_memmap(memmap_dir, manifest["fields"]["image_head_stale"])

    hands = ("left", "right")
    shard_counts: dict[tuple[int, str], np.ndarray] = {
        (split_id, hand): np.zeros((len(episode_id),), dtype=np.int64)
        for split_id in range(len(split_names))
        for hand in hands
    }
    shard_handles = {}
    shard_frame_files: dict[tuple[int, str], str] = {}
    for split_id, split_name in enumerate(split_names):
        split_safe = _sanitize_index_name(split_name)
        for hand in hands:
            key = (split_id, hand)
            frame_file = f"split{split_id}_{split_safe}_{hand}_frame_idx.bin"
            shard_frame_files[key] = frame_file
            shard_handles[key] = (output_dir / frame_file).open("wb")

    started = time.perf_counter()
    try:
        for ep_idx, (start, end) in enumerate(zip(episode_start_idx, episode_end_idx)):
            ep_started = time.perf_counter()
            for chunk_start in range(int(start), int(end), int(chunk_size)):
                chunk_end = min(chunk_start + int(chunk_size), int(end))
                tracked = np.asarray(tracked_mm[chunk_start:chunk_end], dtype=bool)
                stale = np.asarray(stale_mm[chunk_start:chunk_end], dtype=bool)
                label_valid = np.asarray(label_valid_mm[chunk_start:chunk_end], dtype=bool)
                split_ids = np.asarray(frame_split_mm[chunk_start:chunk_end], dtype=np.int32).ravel()
                base = tracked & (~stale)
                if not np.any(base):
                    continue

                for split_id in np.unique(split_ids[base]).tolist():
                    split_mask = base & (split_ids == int(split_id))
                    if not np.any(split_mask):
                        continue
                    for hand_idx, hand in enumerate(hands):
                        local = np.nonzero(split_mask & label_valid[:, hand_idx])[0]
                        if len(local) == 0:
                            continue
                        frames = (chunk_start + local).astype(np.int64, copy=False)
                        frames.tofile(shard_handles[(int(split_id), hand)])
                        shard_counts[(int(split_id), hand)][ep_idx] += int(frames.shape[0])
            if log_every_episode:
                print(
                    "[egoemg_vision_index] "
                    f"episode {ep_idx + 1}/{len(episode_id)} "
                    f"frames={int(end - start)} elapsed_s={time.perf_counter() - ep_started:.2f}",
                    flush=True,
                )
    finally:
        for handle in shard_handles.values():
            handle.close()

    shards: dict[str, dict[str, Any]] = {}
    for split_id, split_name in enumerate(split_names):
        split_safe = _sanitize_index_name(split_name)
        for hand in hands:
            key = f"split{split_id}_{split_safe}_{hand}"
            counts = shard_counts[(split_id, hand)]
            offsets = np.concatenate(
                [np.zeros((1,), dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
            )
            frame_file = shard_frame_files[(split_id, hand)]
            offsets_file = f"{key}_episode_offsets.npy"
            np.save(output_dir / offsets_file, offsets)
            shards[f"{split_id}:{hand}"] = {
                "split_id": split_id,
                "split_name": split_name,
                "hand": hand,
                "frame_idx_file": frame_file,
                "frame_idx_dtype": "int64",
                "episode_offsets_file": offsets_file,
                "num_samples": int(offsets[-1]),
            }

    index_manifest = {
        "version": VISION_INDEX_VERSION,
        "memmap_dir": str(memmap_dir.resolve()),
        "num_episodes": len(episode_id),
        "episode_id": episode_id,
        "split_names": split_names,
        "hands": list(hands),
        "chunk_size": int(chunk_size),
        "created_elapsed_s": time.perf_counter() - started,
        "shards": shards,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(index_manifest, f, indent=2)
    print(
        "[egoemg_vision_index] "
        f"wrote {output_dir} elapsed_s={index_manifest['created_elapsed_s']:.2f}",
        flush=True,
    )
    return output_dir


class EgoEmgVisionIndex:
    """Precomputed valid-sample index for EgoEmgVisionDataset."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        with (self.index_dir / "manifest.json").open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        if int(self.manifest.get("version", -1)) != VISION_INDEX_VERSION:
            raise ValueError(
                f"Unsupported EgoEMG vision index version: {self.manifest.get('version')}"
            )
        self.split_names = list(self.manifest["split_names"])
        self._frame_cache: dict[tuple[int, str], np.ndarray] = {}
        self._offset_cache: dict[tuple[int, str], np.ndarray] = {}

    def _load_shard(self, split_id: int, hand: str) -> tuple[np.ndarray, np.ndarray]:
        key = (int(split_id), hand)
        if key not in self._frame_cache:
            import time as _t
            _t0 = _t.perf_counter()
            shard = self.manifest["shards"].get(f"{split_id}:{hand}")
            if shard is None:
                raise KeyError(f"Missing vision index shard split={split_id} hand={hand}")
            print(f"  [_load_shard] mapping {shard['frame_idx_file']} ({shard['num_samples']} samples)...", flush=True)
            self._frame_cache[key] = np.memmap(
                self.index_dir / shard["frame_idx_file"],
                dtype=np.dtype(shard.get("frame_idx_dtype", "int64")),
                mode="r",
                shape=(int(shard["num_samples"]),),
            )
            _t1 = _t.perf_counter()
            print(f"  [_load_shard] memmap created in {_t1-_t0:.2f}s, loading offsets...", flush=True)
            self._offset_cache[key] = np.load(
                self.index_dir / shard["episode_offsets_file"],
                mmap_mode="r",
            )
            print(f"  [_load_shard] shard {split_id}:{hand} ready in {_t.perf_counter()-_t0:.2f}s", flush=True)
        return self._frame_cache[key], self._offset_cache[key]

    def _select_frames_for_hand(
        self,
        *,
        ep_idx: int,
        hand: str,
        split_ids: Sequence[int],
        stride: int,
        limit: int | None,
    ) -> np.ndarray:
        segments: list[np.ndarray] = []
        for split_id in split_ids:
            frame_idx, offsets = self._load_shard(int(split_id), hand)
            start = int(offsets[ep_idx])
            end = int(offsets[ep_idx + 1])
            if end <= start:
                continue
            segments.append(frame_idx[start:end])

        if not segments:
            return np.zeros((0,), dtype=np.int64)

        if len(segments) == 1:
            frames = np.asarray(segments[0][::stride], dtype=np.int64)
            if limit is not None:
                frames = frames[:limit]
            return frames

        stop = None if limit is None else int(limit) * int(stride)
        sampled_iter = islice(heapq_merge(*segments), 0, stop, int(stride))
        count = -1 if limit is None else int(limit)
        return np.fromiter(sampled_iter, dtype=np.int64, count=count)

    def select(
        self,
        *,
        episode_ids: np.ndarray,
        hand_ids: Sequence[int],
        split_ids: Sequence[int],
        stride: int,
        limit: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        episode_out: list[np.ndarray] = []
        frame_out: list[np.ndarray] = []
        hand_out: list[np.ndarray] = []
        remaining = None if limit is None else int(limit)

        hand_names = {0: "left", 1: "right"}
        for ep_idx in episode_ids.tolist():
            per_hand: dict[int, np.ndarray] = {}
            for hand_idx in hand_ids:
                frames = self._select_frames_for_hand(
                    ep_idx=int(ep_idx),
                    hand=hand_names[int(hand_idx)],
                    split_ids=split_ids,
                    stride=int(stride),
                    limit=remaining,
                )
                if frames.size > 0:
                    per_hand[hand_idx] = frames

            if not per_hand:
                continue
            if len(per_hand) == 1:
                hand_idx = next(iter(per_hand))
                frames = per_hand[hand_idx]
                if remaining is not None:
                    frames = frames[:remaining]
                if frames.size == 0:
                    continue
                frame_out.append(frames.astype(np.int64, copy=False))
                episode_out.append(np.full(frames.shape, int(ep_idx), dtype=np.int32))
                hand_out.append(np.full(frames.shape, int(hand_idx), dtype=np.int8))
                if remaining is not None:
                    remaining -= int(frames.shape[0])
            else:
                hids = sorted(per_hand.keys())
                first_hand, second_hand = hids
                first_frames = per_hand[first_hand]
                second_frames = per_hand[second_hand]
                min_len = min(len(first_frames), len(second_frames))
                total_len = len(first_frames) + len(second_frames)
                all_frames = np.empty((total_len,), dtype=np.int64)
                all_hands = np.empty((total_len,), dtype=np.int8)

                paired = 2 * min_len
                all_frames[0:paired:2] = first_frames[:min_len]
                all_frames[1:paired:2] = second_frames[:min_len]
                all_hands[0:paired:2] = first_hand
                all_hands[1:paired:2] = second_hand

                tail = paired
                if len(first_frames) > min_len:
                    extra = first_frames[min_len:]
                    all_frames[tail:tail + len(extra)] = extra
                    all_hands[tail:tail + len(extra)] = first_hand
                    tail += len(extra)
                if len(second_frames) > min_len:
                    extra = second_frames[min_len:]
                    all_frames[tail:tail + len(extra)] = extra
                    all_hands[tail:tail + len(extra)] = second_hand

                if remaining is not None:
                    all_frames = all_frames[:remaining]
                    all_hands = all_hands[:remaining]
                if all_frames.size == 0:
                    continue
                frame_out.append(all_frames.astype(np.int64, copy=False))
                episode_out.append(np.full(all_frames.shape, int(ep_idx), dtype=np.int32))
                hand_out.append(all_hands)
                if remaining is not None:
                    remaining -= int(all_frames.shape[0])

            if remaining is not None and remaining <= 0:
                break

        if not frame_out:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int8),
            )
        return (
            np.concatenate(episode_out).astype(np.int32, copy=False),
            np.concatenate(frame_out).astype(np.int64, copy=False),
            np.concatenate(hand_out).astype(np.int8, copy=False),
        )


def _rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(rotmat.astype(np.float64))
    return rvec.reshape(3).astype(np.float32)


def _expand_to_aspect_ratio(
    input_shape: np.ndarray,
    target_aspect_ratio: tuple[int, int] = (192, 256),
) -> np.ndarray:
    w, h = input_shape
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        return np.array([w, max(w * h_t / w_t, h)], dtype=np.float32)
    return np.array([max(h * w_t / h_t, w), h], dtype=np.float32)


def _get_bbox(keypoints_2d: np.ndarray, rescale: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    valid = keypoints_2d[:, -1] > 0
    valid_keypoints = keypoints_2d[valid][:, :-1]
    center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0)) / 2
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    scale = bbox_size * rescale
    return center.astype(np.float32), scale.astype(np.float32)


def _rotate_2d(pt_2d: np.ndarray, rot_rad: float) -> np.ndarray:
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)


def _gen_trans_from_patch_cv(
    c_x: float,
    c_y: float,
    src_width: float,
    src_height: float,
    dst_width: float,
    dst_height: float,
    scale: float,
    rot: float,
) -> np.ndarray:
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180.0
    src_downdir = _rotate_2d(np.array([0.0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = _rotate_2d(np.array([src_w * 0.5, 0.0], dtype=np.float32), rot_rad)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0.0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0.0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _trans_point2d(pt_2d: np.ndarray, trans: np.ndarray) -> np.ndarray:
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.0], dtype=np.float32)
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def _generate_image_patch_cv2(
    img: np.ndarray,
    c_x: float,
    c_y: float,
    bb_width: float,
    bb_height: float,
    patch_width: float,
    patch_height: float,
    do_flip: bool,
    scale: float,
    rot: float,
    border_mode: int = cv2.BORDER_CONSTANT,
) -> tuple[np.ndarray, np.ndarray]:
    img_width = img.shape[1]
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1
    trans = _gen_trans_from_patch_cv(
        c_x,
        c_y,
        bb_width,
        bb_height,
        patch_width,
        patch_height,
        scale,
        rot,
    )
    patch = cv2.warpAffine(
        img,
        trans,
        (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0,
    )
    return patch, trans


def _convert_cvimg_to_tensor(cvimg: np.ndarray) -> np.ndarray:
    return np.transpose(cvimg, (2, 0, 1)).astype(np.float32)


def _fliplr_keypoints(joints: np.ndarray, width: float, flip_permutation: list[int]) -> np.ndarray:
    joints = joints.copy()
    joints[:, 0] = width - joints[:, 0] - 1
    return joints[flip_permutation, :]


def _keypoint_3d_processing(
    keypoints_3d: np.ndarray,
    flip_permutation: list[int],
    rot: float,
    do_flip: bool,
) -> np.ndarray:
    keypoints_3d = keypoints_3d.copy()
    if do_flip:
        keypoints_3d = _fliplr_keypoints(keypoints_3d, 1, flip_permutation)
    rot_mat = np.eye(3, dtype=np.float32)
    if rot != 0:
        rot_rad = -rot * np.pi / 180.0
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    keypoints_3d[:, :-1] = np.einsum("ij,kj->ki", rot_mat, keypoints_3d[:, :-1])
    return keypoints_3d.astype(np.float32)


def _rot_aa(aa: np.ndarray, rot: float) -> np.ndarray:
    rotm = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    per_rdg, _ = cv2.Rodrigues(aa.astype(np.float64))
    resrot, _ = cv2.Rodrigues(np.dot(rotm, per_rdg))
    return resrot.reshape(3).astype(np.float32)


def _mano_param_processing(
    mano_params: dict[str, np.ndarray],
    has_mano_params: dict[str, np.bool_],
    rot: float,
    do_flip: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.bool_]]:
    mano_params = {k: v.copy() for k, v in mano_params.items()}
    has_mano_params = dict(has_mano_params)
    if do_flip:
        mano_params["global_orient"][1::3] *= -1
        mano_params["global_orient"][2::3] *= -1
        mano_params["hand_pose"][1::3] *= -1
        mano_params["hand_pose"][2::3] *= -1
    mano_params["global_orient"] = _rot_aa(mano_params["global_orient"], rot)
    return mano_params, has_mano_params


def _do_augmentation(aug_config: Any) -> tuple[float, float, bool, list[float], float, float]:
    tx = np.clip(np.random.randn(), -1.0, 1.0) * float(aug_config.TRANS_FACTOR)
    ty = np.clip(np.random.randn(), -1.0, 1.0) * float(aug_config.TRANS_FACTOR)
    scale = np.clip(np.random.randn(), -1.0, 1.0) * float(aug_config.SCALE_FACTOR) + 1.0
    rot = (
        np.clip(np.random.randn(), -2.0, 2.0) * float(aug_config.ROT_FACTOR)
        if random.random() <= float(aug_config.ROT_AUG_RATE)
        else 0.0
    )
    do_flip = bool(getattr(aug_config, "DO_FLIP", False)) and random.random() <= float(
        getattr(aug_config, "FLIP_AUG_RATE", 0.0)
    )
    color_scale = []
    c_up = 1.0 + float(aug_config.COLOR_SCALE)
    c_low = 1.0 - float(aug_config.COLOR_SCALE)
    for _ in range(3):
        color_scale.append(random.uniform(c_low, c_up))
    return scale, rot, do_flip, color_scale, tx, ty


def _world_to_camera(points_world: np.ndarray, T_W_C: np.ndarray) -> np.ndarray:
    T_C_W = np.linalg.inv(T_W_C)
    R_C_W = T_C_W[:3, :3]
    t_C_W = T_C_W[:3, 3]
    return (R_C_W @ points_world.T).T + t_C_W


def _project_world_points(
    points_world_xyz: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T_C_W = np.linalg.inv(T_W_C)
    R_C_W = T_C_W[:3, :3]
    t_C_W = T_C_W[:3, 3].reshape(3, 1)

    p_cam = (R_C_W @ points_world_xyz.T + t_C_W).T
    depth_pos = p_cam[:, 2] > 1e-6

    rvec, _ = cv2.Rodrigues(R_C_W)
    proj, _ = cv2.projectPoints(
        points_world_xyz.astype(np.float64),
        rvec,
        t_C_W,
        K,
        dist,
    )
    return proj.reshape(-1, 2), depth_pos


def _map_processed_points_to_raw(
    points_proc: np.ndarray,
    intrinsics_info: dict[str, Any],
) -> np.ndarray:
    if str(intrinsics_info.get("mode", "none")) != "gopro_8x7_crop_upsample":
        return points_proc.copy()

    crop_xywh = intrinsics_info.get("crop_xywh_on_video")
    processed_size = intrinsics_info.get("processed_size")
    if crop_xywh is None or processed_size is None:
        return points_proc.copy()

    x0, y0, crop_w, crop_h = [float(v) for v in crop_xywh]
    proc_w, proc_h = [float(v) for v in processed_size]
    out = points_proc.astype(np.float64).copy()
    out[:, 0] = (out[:, 0] / proc_w) * crop_w + x0
    out[:, 1] = (out[:, 1] / proc_h) * crop_h + y0
    return out


def _extract_crop_params(
    intrinsics_info: dict[str, Any],
    video_w: int,
    video_h: int,
) -> np.ndarray:
    crop_xywh = intrinsics_info.get(
        "crop_xywh_on_video", [0, 0, video_w, video_h],
    )
    proc_size = intrinsics_info.get("processed_size", [video_w, video_h])
    return np.array([
        float(proc_size[0]), float(proc_size[1]),
        float(crop_xywh[2]), float(crop_xywh[3]),
        float(crop_xywh[0]), float(crop_xywh[1]),
    ], dtype=np.float32)


def _build_intrinsics_and_frame_mapper(
    K: np.ndarray,
    dist: np.ndarray,
    calib_w: int,
    calib_h: int,
    video_w: int,
    video_h: int,
    first_frame_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # EgoEMG head-view videos are 1280x720 with side bars. WiLoR supervision needs
    # projections in raw video pixels, so we project using the processed calib
    # frame and map back to raw video coordinates.
    gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY)
    col_mean = gray.mean(axis=0)
    active_cols = np.where(col_mean > 2.0)[0]
    if len(active_cols) > 0:
        x0 = int(active_cols[0])
        x1 = int(active_cols[-1]) + 1
    else:
        active_w = int(round(video_h * (calib_w / float(calib_h))))
        x0 = int((video_w - active_w) // 2)
        x1 = x0 + active_w

    active_w = x1 - x0
    info = {
        "mode": "gopro_8x7_crop_upsample",
        "crop_xywh_on_video": [x0, 0, active_w, video_h],
        "processed_size": [calib_w, calib_h],
    }
    return K.copy(), dist.copy(), info


# Shared read-only resources across dataset instances with the same memmap_dir.
_ds_resource_cache: dict[str, dict] = {}


@dataclass
class EgoEmgVisionDataset(Dataset):
    """Expose EgoEMG as WiLoR-native training samples."""

    memmap_dir: Path
    video_root: Path | None = None
    allintra_root: Path | None = None
    allintra_suffix: str = DEFAULT_ALLINTRA_SUFFIX
    calibration_path: Path | None = None
    # When False, skip loading per-episode camera calibration and fall back to
    # an identity intrinsics + no-distortion default (so headless visualization
    # can render without the reprojection_assets calibration files).
    load_calibration: bool = True
    allowed_episode_ids: Sequence[str] | None = None
    allowed_subjects: Sequence[str] | None = None
    allowed_splits: Sequence[str] | None = None
    target_hand: str = "both"
    stride: int = 1
    index_limit: int | None = None
    vision_index_dir: Path | None = None
    auto_build_index: bool = False
    patch_size: int = 256
    return_frame_bgr: bool = False
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    aug_config: Any | None = None
    do_augment: bool = False
    index_chunk_size: int = 1_000_000
    frame_cache_size: int = 2
    video_reader_cache_size: int = 4
    video_gpu_id: int | None = None
    log_init_timing: bool = False
    per_episode_crops_dir: Path | None = None
    cached_vit_features_dir: Path | None = None

    def __post_init__(self) -> None:
        t_init = time.perf_counter()
        self.memmap_dir = Path(self.memmap_dir)
        self.video_root = Path(self.video_root) if self.video_root else self.memmap_dir.parent
        self.vision_index_dir = (
            Path(self.vision_index_dir)
            if self.vision_index_dir is not None
            else _default_vision_index_dir(self.memmap_dir)
        )
        self.allintra_root = (
            Path(self.allintra_root)
            if self.allintra_root is not None
            else self.video_root.parent / "EgoEMG_allintra"
        )
        self.calibration_path = (
            Path(self.calibration_path)
            if self.calibration_path is not None
            else self.video_root / "reprojection_assets" / "GX010023_standard_calibration.json"
        )
        if self.target_hand not in {"left", "right", "both"}:
            raise ValueError(f"target_hand must be left/right/both, got {self.target_hand}")

        self.mean = (
            np.asarray(self.mean, dtype=np.float32)
            if self.mean is not None
            else 255.0 * np.array([0.485, 0.456, 0.406], dtype=np.float32)
        )
        self.std = (
            np.asarray(self.std, dtype=np.float32)
            if self.std is not None
            else 255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32)
        )

        # Per-instance runtime state (not shared)
        self._video_cache = self._make_video_cache()
        self._frame_cache: dict[tuple[str, int], np.ndarray] = {}
        self._intrinsics_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        self._runtime_pid = os.getpid()

        cache_key = str(self.memmap_dir.resolve())
        if cache_key in _ds_resource_cache:
            t = time.perf_counter()
            self._attach_cached(_ds_resource_cache[cache_key])
            self._log_init_stage("attach_cache", t)
        else:
            t = time.perf_counter()
            with (self.memmap_dir / "manifest.json").open("r", encoding="utf-8") as f:
                self._manifest = json.load(f)
            self._log_init_stage("manifest", t)
            t = time.perf_counter()
            metadata = np.load(self.memmap_dir / "metadata.npz", allow_pickle=False)
            self._log_init_stage("metadata", t)
            self._episode_id = _decode_bytes(metadata["episode_id"])
            self._episode_subject = _decode_bytes(metadata["episode_subject"])
            self._episode_subject_id = metadata["episode_subject_id"]
            self._episode_head_video_path = _decode_bytes(metadata["episode_head_video_path"])
            self._episode_start_idx = metadata["episode_start_idx"]
            self._episode_end_idx = metadata["episode_end_idx"]
            self._episode_beta_idx = metadata["episode_beta_idx"]
            self._subjects = _decode_bytes(metadata["subjects_subject"])
            self._splits = _decode_bytes(metadata["splits_split"])

            self._frame_memmaps: dict[str, np.memmap] = {}
            self._episode_memmaps: dict[str, np.memmap] = {}

            t = time.perf_counter()
            self._open_data_memmaps()
            self._log_init_stage("open_memmaps", t)

            t = time.perf_counter()
            if self.load_calibration and self.calibration_path is not None:
                try:
                    with self.calibration_path.open("r", encoding="utf-8") as f:
                        calib = json.load(f)
                    self._K_calib = np.asarray(calib["camera_matrix"], dtype=np.float64)
                    self._dist_calib = np.asarray(
                        calib["distortion_coefficients"],
                        dtype=np.float64,
                    ).reshape(-1, 1)
                    self._calib_w = int(calib["image_width"])
                    self._calib_h = int(calib["image_height"])
                except (OSError, KeyError, ValueError) as exc:
                    # Missing / malformed per-episode calibration: fall back to an
                    # identity intrinsics + no-distortion default so headless
                    # visualization can render without the calibration assets.
                    self._K_calib = np.eye(3, dtype=np.float64)
                    self._dist_calib = np.zeros((1, 5), dtype=np.float64)
                    self._calib_w = 640
                    self._calib_h = 480
                    self._calib_skipped = True
                    print(
                        f"Warning: calibration unavailable ({exc}); using identity intrinsics.",
                        flush=True,
                    )
            else:
                # Identity intrinsics + no distortion; enables headless rendering
                # when per-episode calibration assets are unavailable.
                self._K_calib = np.eye(3, dtype=np.float64)
                self._dist_calib = np.zeros((1, 5), dtype=np.float64)
                self._calib_w = 640
                self._calib_h = 480
            self._log_init_stage("calibration", t)

            _ds_resource_cache[cache_key] = self._cacheable()

        if self.per_episode_crops_dir is not None:
            t = time.perf_counter()
            self._init_per_episode_crops()
            self._log_init_stage("per_episode_crops_index", t)
        else:
            t = time.perf_counter()
            self._build_index()
            self._log_init_stage("build_index", t)

        if self.cached_vit_features_dir is not None:
            t = time.perf_counter()
            self._init_cached_vit_features()
            self._log_init_stage("cached_vit_features_index", t)

        self._log_init_stage("total", t_init)

    def _init_per_episode_crops(self) -> None:
        """Build sample index from vision_index, then open per-episode LMDBs on demand."""

        crops_dir = Path(self.per_episode_crops_dir)
        manifest_path = crops_dir / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                self._pec_manifest = json.load(f)
            manifest_ps = int(self._pec_manifest.get("patch_size", self.patch_size))
            if manifest_ps != int(self.patch_size):
                raise ValueError(
                    f"Per-episode crops patch_size={manifest_ps} != dataset patch_size={self.patch_size}"
                )
            intrinsics = self._pec_manifest.get("intrinsics", {})
            self._pec_video_w = int(intrinsics.get("video_w", 1280))
            self._pec_video_h = int(intrinsics.get("video_h", 720))
            self._pec_K = np.asarray(intrinsics["K"], dtype=np.float64)
            self._pec_dist = np.asarray(intrinsics["dist"], dtype=np.float64)
            self._pec_intrinsics_info = intrinsics["info"]
        else:
            raise FileNotFoundError(f"Missing per-episode crops manifest at {manifest_path}")

        self._build_index()

        self._pec_lmdb_cache: dict[int, tuple] = {}
        self._pec_lmdb_max_open = 8
        self._pec_pid = os.getpid()

    def _get_pec_txn(self, ep_idx: int):
        """Get or open an LMDB read transaction for a per-episode crops database."""
        import lmdb as _lmdb  # noqa: F811

        if os.getpid() != self._pec_pid:
            self._pec_lmdb_cache = {}
            self._pec_pid = os.getpid()

        cached = self._pec_lmdb_cache.get(ep_idx)
        if cached is not None:
            return cached[1]

        if len(self._pec_lmdb_cache) >= self._pec_lmdb_max_open:
            oldest_key = next(iter(self._pec_lmdb_cache))
            old_env, _ = self._pec_lmdb_cache.pop(oldest_key)
            old_env.close()

        ep_id = self._episode_id[ep_idx]
        lmdb_path = Path(self.per_episode_crops_dir) / f"{ep_id}.lmdb"
        if not lmdb_path.exists():
            return None
        env = _lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
        txn = env.begin()
        self._pec_lmdb_cache[ep_idx] = (env, txn)
        return txn

    # ── Cached ViT features (per-episode LMDB, mirrors crops layout) ──────

    def _init_cached_vit_features(self) -> None:
        """Open per-episode ViT feature LMDBs on demand.

        Feature LMDBs use the same key scheme as crops LMDBs:
        {frame_idx:08d}_{L/R} -> 1280 float32 bytes.
        No manifest, no split directories, no index arrays needed.
        """
        self._cvf_dir = Path(self.cached_vit_features_dir)
        self._cvf_lmdb_cache: dict[int, tuple] = {}
        self._cvf_lmdb_max_open = 8
        self._cvf_pid = os.getpid()

    def _get_cvf_txn(self, ep_idx: int):
        """Get or open an LMDB read transaction for a per-episode feature database."""
        import lmdb as _lmdb  # noqa: F811

        if os.getpid() != self._cvf_pid:
            self._cvf_lmdb_cache = {}
            self._cvf_pid = os.getpid()

        cached = self._cvf_lmdb_cache.get(ep_idx)
        if cached is not None:
            return cached[1]

        if len(self._cvf_lmdb_cache) >= self._cvf_lmdb_max_open:
            oldest_key = next(iter(self._cvf_lmdb_cache))
            old_env, _ = self._cvf_lmdb_cache.pop(oldest_key)
            old_env.close()

        ep_id = self._episode_id[ep_idx]
        lmdb_path = self._cvf_dir / f"{ep_id}.lmdb"
        if not lmdb_path.exists():
            return None
        env = _lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
        txn = env.begin()
        self._cvf_lmdb_cache[ep_idx] = (env, txn)
        return txn

    def _make_video_cache(self):
        if self.video_gpu_id is not None:
            return FfmpegNvdecReaderCache(
                gpu_id=self.video_gpu_id, max_readers=self.video_reader_cache_size
            )
        return DecordVideoReaderCache(max_readers=self.video_reader_cache_size)

    def _cacheable(self) -> dict:
        return {
            "_manifest": self._manifest,
            "_episode_id": self._episode_id,
            "_episode_subject": self._episode_subject,
            "_episode_subject_id": self._episode_subject_id,
            "_episode_head_video_path": self._episode_head_video_path,
            "_episode_start_idx": self._episode_start_idx,
            "_episode_end_idx": self._episode_end_idx,
            "_episode_beta_idx": self._episode_beta_idx,
            "_subjects": self._subjects,
            "_splits": self._splits,
            "_frame_memmaps": self._frame_memmaps,
            "_episode_memmaps": self._episode_memmaps,
            "_K_calib": self._K_calib,
            "_dist_calib": self._dist_calib,
            "_calib_w": self._calib_w,
            "_calib_h": self._calib_h,
        }

    def _attach_cached(self, cached: dict) -> None:
        for k, v in cached.items():
            setattr(self, k, v)

    def _log_init_stage(self, name: str, start_time: float) -> None:
        if not self.log_init_timing:
            return
        elapsed = time.perf_counter() - start_time
        print(f"[egoemg_vision_dataset:init] {name} elapsed_s={elapsed:.3f}", flush=True)

    def _resolve_allowed_split_ids(self) -> set[int] | None:
        if not self.allowed_splits:
            return None
        return _resolve_str_filter(self._splits, self.allowed_splits, "split")

    def _build_index(self) -> None:
        allowed_split_ids = self._resolve_allowed_split_ids()

        n_ep = len(self._episode_id)
        ep_mask = np.ones((n_ep,), dtype=bool)
        if self.allowed_episode_ids:
            allowed = set(self.allowed_episode_ids)
            ep_mask &= np.asarray([eid in allowed for eid in self._episode_id], dtype=bool)
        if self.allowed_subjects:
            allowed_subject_ids = _resolve_str_filter(
                self._subjects,
                self.allowed_subjects,
                "subject",
            )
            ep_mask &= np.isin(self._episode_subject_id, list(allowed_subject_ids))
        allowed_eps = np.nonzero(ep_mask)[0]

        hands = [0, 1]
        if self.target_hand == "left":
            hands = [0]
        elif self.target_hand == "right":
            hands = [1]

        if not (self.vision_index_dir / "manifest.json").exists():
            if not self.auto_build_index:
                raise FileNotFoundError(
                    "Missing EgoEMG vision index. Build it once with "
                    "`python scripts/build_egoemg_vision_index.py "
                    f"--memmap-dir {self.memmap_dir} "
                    f"--output-dir {self.vision_index_dir}` "
                    "or pass auto_build_index=True for one-time migration."
                )
            build_egoemg_vision_index(
                memmap_dir=self.memmap_dir,
                output_dir=self.vision_index_dir,
                chunk_size=self.index_chunk_size,
            )

        vision_index = EgoEmgVisionIndex(self.vision_index_dir)
        split_ids = (
            list(range(len(self._splits)))
            if allowed_split_ids is None
            else sorted(allowed_split_ids)
        )
        (
            self._sample_episode_idx,
            self._sample_frame_idx,
            self._sample_hand,
        ) = vision_index.select(
            episode_ids=allowed_eps,
            hand_ids=hands,
            split_ids=split_ids,
            stride=int(self.stride),
            limit=self.index_limit,
        )
        # Downcast frame index to int32 to save memory per worker.
        if self._sample_frame_idx.size > 0:
            self._sample_frame_idx = self._sample_frame_idx.astype(np.int32, copy=False)

    def __len__(self) -> int:
        return len(self._sample_frame_idx)

    def _open_data_memmaps(self) -> None:
        self._frame_memmaps = {}
        self._episode_memmaps = {}
        for group_name in ("mano", "joint_angles", "mocap_hands", "camera", "video_index", "split"):
            for field_name in MODALITY_GROUPS[group_name]:
                self._frame_memmaps[field_name] = _load_memmap(
                    self.memmap_dir,
                    self._manifest["fields"][field_name],
                )
        for field_name in ("generated_mano_left_beta", "generated_mano_right_beta"):
            self._episode_memmaps[field_name] = _load_memmap(
                self.memmap_dir,
                self._manifest["episode_fields"][field_name],
            )

    def reset_runtime_caches(self) -> None:
        """Drop process-local native caches before DataLoader workers fork."""
        self._video_cache = self._make_video_cache()
        self._frame_cache.clear()
        self._intrinsics_cache.clear()
        self._runtime_pid = os.getpid()

    def _ensure_runtime_owner(self) -> None:
        if self._runtime_pid != os.getpid():
            self.reset_runtime_caches()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_frame_memmaps"] = {}
        state["_episode_memmaps"] = {}
        state["_video_cache"] = None
        state["_frame_cache"] = {}
        state["_intrinsics_cache"] = {}
        state.pop("_pec_lmdb_cache", None)
        state.pop("_cvf_lmdb_cache", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_data_memmaps()
        self._video_cache = self._make_video_cache()
        self._frame_cache = {}
        self._intrinsics_cache = {}
        self._runtime_pid = os.getpid()
        if self.per_episode_crops_dir is not None:
            self._pec_lmdb_cache = {}
            self._pec_pid = os.getpid()
        if self.cached_vit_features_dir is not None:
            self._cvf_lmdb_cache = {}
            self._cvf_pid = os.getpid()

    def _resolve_video_path(self, raw_video_path: str) -> str:
        return str(
            resolve_allintra_video_path(
                raw_video_path=raw_video_path,
                data_root=self.video_root,
                allintra_root=self.allintra_root,
                suffix=self.allintra_suffix,
            )
        )

    def _read_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        key = (video_path, int(frame_idx))
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached.copy()
        frame_rgb = self._video_cache.read_one_rgb(video_path, int(frame_idx))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if len(self._frame_cache) >= self.frame_cache_size:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[key] = frame_bgr
        return frame_bgr.copy()

    def _get_intrinsics(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        video_h, video_w = frame_bgr.shape[:2]
        key = (video_w, video_h)
        cached = self._intrinsics_cache.get(key)
        if cached is not None:
            return cached
        cached = _build_intrinsics_and_frame_mapper(
            self._K_calib,
            self._dist_calib,
            self._calib_w,
            self._calib_h,
            video_w,
            video_h,
            frame_bgr,
        )
        self._intrinsics_cache[key] = cached
        return cached

    # ── Label reading helpers ─────────────────────────────────────────────

    def _read_memmap_labels(
        self, frame_idx: int, ep_idx: int, hand_idx: int,
    ) -> dict[str, Any]:
        """Read all memmap-derived labels for one sample.

        Returns dict with hand identity, camera transforms, MANO params,
        joint angles, and marker data.
        """
        hand = "left" if hand_idx == 0 else "right"
        is_left = hand == "left"
        video_frame_idx = int(
            self._frame_memmaps["image_head_frame_index"][frame_idx]
        )

        T_W_C = np.eye(4, dtype=np.float64)
        t12_cam = np.asarray(
            self._frame_memmaps["mocap_head_transform"][frame_idx],
            dtype=np.float64,
        )
        T_W_C[:3, :3] = t12_cam[:9].reshape(3, 3)
        T_W_C[:3, 3] = t12_cam[9:12]

        marker_world = np.asarray(
            self._frame_memmaps[f"mocap_{hand}_keypoints"][frame_idx],
            dtype=np.float64,
        )
        marker_valid = np.asarray(
            self._frame_memmaps[f"mocap_{hand}_valid"][frame_idx], dtype=bool
        )
        label_valid = bool(
            self._frame_memmaps["generated_label_valid"][frame_idx, hand_idx]
        )

        beta_idx = int(self._episode_beta_idx[ep_idx])
        beta = np.asarray(
            self._episode_memmaps[f"generated_mano_{hand}_beta"][beta_idx],
            dtype=np.float32,
        ).copy()
        mano_pose = np.asarray(
            self._frame_memmaps[f"generated_mano_{hand}_pose"][frame_idx],
            dtype=np.float32,
        ).copy()
        hand_pose = mano_pose[3:].copy()

        ja_raw = np.asarray(
            self._frame_memmaps[f"generated_joint_angles_{hand}"][frame_idx],
            dtype=np.float32,
        ).copy()
        wrist_pitch = np.deg2rad(np.asarray(
            self._frame_memmaps[f"mocap_{hand}_wrist_pitch"][frame_idx],
            dtype=np.float32,
        ))
        wrist_yaw = np.deg2rad(np.asarray(
            self._frame_memmaps[f"mocap_{hand}_wrist_yaw"][frame_idx],
            dtype=np.float32,
        ))
        joint_angles = np.concatenate(
            [ja_raw, [wrist_pitch, wrist_yaw]]
        ).astype(np.float32)

        t12_world = np.asarray(
            self._frame_memmaps[f"mocap_mano_{hand}_world_transform"][frame_idx],
            dtype=np.float64,
        )
        R_world = t12_world[:9].reshape(3, 3)
        T_C_W = np.linalg.inv(T_W_C)
        R_C_W = T_C_W[:3, :3]
        t_world = t12_world[9:12]
        wrist_pos_cam = (R_C_W @ t_world + T_C_W[:3, 3]).astype(np.float32)
        global_rot = R_C_W @ R_world
        if is_left:
            global_rot = MIRROR_X_3 @ global_rot @ MIRROR_X_3
            wrist_pos_cam[0] *= -1.0
        global_orient = _rotmat_to_axis_angle(global_rot)

        return {
            "hand": hand,
            "is_left": is_left,
            "video_frame_idx": video_frame_idx,
            "T_W_C": T_W_C,
            "marker_world": marker_world,
            "marker_valid": marker_valid,
            "label_valid": label_valid,
            "joint_angles": joint_angles,
            "wrist_pos_cam": wrist_pos_cam,
            "mano_params": {
                "global_orient": global_orient,
                "hand_pose": hand_pose.astype(np.float32),
                "betas": beta.astype(np.float32),
            },
            "has_mano_params": {
                "global_orient": np.bool_(label_valid),
                "hand_pose": np.bool_(label_valid),
                "betas": np.bool_(label_valid),
            },
            "mano_params_is_axis_angle": {
                "global_orient": np.bool_(True),
                "hand_pose": np.bool_(True),
                "betas": np.bool_(False),
            },
        }

    def _project_keypoints_and_bbox(
        self,
        labels: dict[str, Any],
        K: np.ndarray,
        dist: np.ndarray,
        intrinsics_info: dict[str, Any],
        video_w: int,
        video_h: int,
    ) -> dict[str, Any]:
        """Project world markers to 2D/3D, mirror for left hand, compute bbox."""
        is_left = labels["is_left"]
        T_W_C = labels["T_W_C"]
        label_valid = labels["label_valid"]

        marker_proc, marker_depth_valid = _project_world_points(
            labels["marker_world"], T_W_C, K, dist
        )
        marker_raw = _map_processed_points_to_raw(marker_proc, intrinsics_info)
        marker_in_image = (
            (marker_raw[:, 0] >= 0)
            & (marker_raw[:, 0] < video_w)
            & (marker_raw[:, 1] >= 0)
            & (marker_raw[:, 1] < video_h)
        )

        markers_2d_xy = marker_raw.astype(np.float32)
        markers_3d_xyz = _world_to_camera(labels["marker_world"], T_W_C).astype(
            np.float32
        )

        if is_left:
            markers_2d_xy[:, 0] = (video_w - 1) - markers_2d_xy[:, 0]
            markers_3d_xyz[:, 0] *= -1.0

        markers_conf = (
            labels["marker_valid"] & marker_depth_valid & marker_in_image
        ).astype(np.float32)
        if not label_valid:
            markers_conf[:] = 0.0

        keypoints_2d = np.concatenate(
            [markers_2d_xy, markers_conf[:, None]], axis=-1
        )
        keypoints_3d = np.concatenate(
            [markers_3d_xyz, markers_conf[:, None]], axis=-1
        )

        valid_for_bbox = keypoints_2d[:, 2] > 0
        if valid_for_bbox.sum() < 2:
            center = np.array(
                [video_w / 2.0, video_h / 2.0], dtype=np.float32
            )
            scale = np.array(
                [video_w / 3.0, video_h / 3.0], dtype=np.float32
            )
            bbox_source_name = "fallback"
        else:
            center, scale = _get_bbox(keypoints_2d, rescale=1.2)
            bbox_source_name = "mocap_markers"
        scale = _expand_to_aspect_ratio(scale, (192, 256))
        bbox_size = float(max(scale[0], scale[1]))

        return {
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "markers_2d": keypoints_2d.copy(),
            "center": center,
            "bbox_size": bbox_size,
            "bbox_source_name": bbox_source_name,
        }

    def _normalize_image(
        self, img_patch_bgr: np.ndarray, color_scale: list[float],
    ) -> np.ndarray:
        """Convert BGR patch to normalized float32 tensor."""
        img_patch = _convert_cvimg_to_tensor(img_patch_bgr[:, :, ::-1])
        for c in range(3):
            img_patch[c, :, :] = np.clip(
                img_patch[c, :, :] * color_scale[c], 0, 255
            )
            img_patch[c, :, :] = (
                img_patch[c, :, :] - self.mean[c]
            ) / self.std[c]
        return img_patch.astype(np.float32)

    def _build_sample(
        self,
        *,
        img_patch: np.ndarray,
        kp2d_patch: np.ndarray,
        kp3d_patch: np.ndarray,
        mano_params_patch: dict[str, np.ndarray],
        has_mano_params_patch: dict[str, np.bool_],
        labels: dict[str, Any],
        kp: dict[str, Any],
        cam_info: dict[str, Any],
        ep_idx: int,
        frame_idx: int,
        video_path: str,
        dataset_name: str,
        frame_bgr: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Construct the final sample dictionary."""
        video_w = cam_info["video_w"]
        video_h = cam_info["video_h"]
        bbox_size = kp["bbox_size"]
        center = kp["center"]

        sample: dict[str, Any] = {
            "img": img_patch,
            "keypoints_2d": kp2d_patch.astype(np.float32),
            "keypoints_3d": kp3d_patch.astype(np.float32),
            "mano_params": {
                "global_orient": mano_params_patch["global_orient"].astype(np.float32),
                "hand_pose": mano_params_patch["hand_pose"].astype(np.float32),
                "betas": mano_params_patch["betas"].astype(np.float32),
            },
            "has_mano_params": {
                "global_orient": np.bool_(has_mano_params_patch["global_orient"]),
                "hand_pose": np.bool_(has_mano_params_patch["hand_pose"]),
                "betas": np.bool_(has_mano_params_patch["betas"]),
            },
            "mano_params_is_axis_angle": labels["mano_params_is_axis_angle"],
            "bbox": np.array([
                center[0] - bbox_size / 2.0, center[1] - bbox_size / 2.0,
                center[0] + bbox_size / 2.0, center[1] + bbox_size / 2.0,
            ], dtype=np.float32),
            "raw_right": np.float32(0.0 if labels["is_left"] else 1.0),
            "img_size": np.array([video_h, video_w], dtype=np.float32),
            "box_center": center.astype(np.float32),
            "box_size": np.float32(bbox_size),
            "bbox_source_name": kp["bbox_source_name"],
            "joint_angles": labels["joint_angles"],
            "label_valid_mask": np.bool_(labels["label_valid"]),
            "orig_keypoints_2d": kp["keypoints_2d"].astype(np.float32),
            "orig_markers_2d": kp["markers_2d"].astype(np.float32),
            "wrist_pos_cam": labels["wrist_pos_cam"],
            "cam_K": cam_info["K"].astype(np.float32),
            "cam_dist": cam_info["dist"].reshape(-1).astype(np.float32),
            "cam_crop_params": cam_info["cam_crop_params"],
            "video_frame_index": np.int64(labels["video_frame_idx"]),
            "video_path": video_path,
            "episode_id": self._episode_id[ep_idx],
            "episode_subject": self._episode_subject[ep_idx],
            "frame_index": np.int64(frame_idx),
            "target_hand": labels["hand"],
            "dataset_name": dataset_name,
        }
        if frame_bgr is not None:
            sample["frame_bgr"] = frame_bgr.astype(np.uint8, copy=False)
        return sample

    # ── __getitem__ paths ─────────────────────────────────────────────────

    def _getitem_cached_vit(self, idx: int) -> dict[str, Any]:
        """Build sample using cached ViT features from per-episode LMDB."""
        ep_idx = int(self._sample_episode_idx[idx])
        frame_idx = int(self._sample_frame_idx[idx])
        hand_idx = int(self._sample_hand[idx])

        labels = self._read_memmap_labels(frame_idx, ep_idx, hand_idx)

        # Look up feature from per-episode LMDB by video_frame_idx + hand key
        hand_letter = "L" if labels["is_left"] else "R"
        lmdb_key = f"{labels['video_frame_idx']:08d}_{hand_letter}"
        txn = self._get_cvf_txn(ep_idx)
        if txn is None:
            raise ValueError(
                f"No feature LMDB for episode {self._episode_id[ep_idx]}"
            )
        feature_bytes = txn.get(lmdb_key.encode("ascii"))
        if feature_bytes is None:
            raise ValueError(
                f"No feature for key {lmdb_key} "
                f"in episode {self._episode_id[ep_idx]}"
            )
        vit_feature = np.frombuffer(feature_bytes, dtype=np.float32)

        sample: dict[str, Any] = {
            "features": torch.from_numpy(vit_feature.copy()),
            "joint_angles": torch.from_numpy(labels["joint_angles"]).clone(),
            "label_valid_mask": np.bool_(labels["label_valid"]),
            "episode_id": self._episode_id[ep_idx],
            "episode_subject": self._episode_subject[ep_idx],
            "target_hand": labels["hand"],
            "dataset_name": "egoemg_cached_vit",
        }
        if get_worker_info() is None:
            self.reset_runtime_caches()
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_runtime_owner()
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if self.cached_vit_features_dir is not None:
            return self._getitem_cached_vit(idx)

        if self.per_episode_crops_dir is not None:
            return self._getitem_per_episode_crops(idx)

        ep_idx = int(self._sample_episode_idx[idx])
        frame_idx = int(self._sample_frame_idx[idx])
        hand_idx = int(self._sample_hand[idx])

        labels = self._read_memmap_labels(frame_idx, ep_idx, hand_idx)
        video_path = self._resolve_video_path(
            self._episode_head_video_path[ep_idx]
        )
        frame_bgr = self._read_frame(video_path, labels["video_frame_idx"])
        K, dist, intrinsics_info = self._get_intrinsics(frame_bgr)
        video_h, video_w = frame_bgr.shape[:2]
        cam_crop_params = _extract_crop_params(intrinsics_info, video_w, video_h)

        kp = self._project_keypoints_and_bbox(
            labels, K, dist, intrinsics_info, video_w, video_h
        )

        if labels["is_left"]:
            frame_bgr = np.ascontiguousarray(frame_bgr[:, ::-1])

        if self.do_augment:
            scale_aug, rot_aug, do_flip, color_scale, tx, ty = (
                _do_augmentation(self.aug_config)
            )
        else:
            scale_aug, rot_aug, do_flip, color_scale, tx, ty = (
                1.0, 0.0, False, [1.0, 1.0, 1.0], 0.0, 0.0
            )

        center_aug = kp["center"].copy()
        center_aug[0] += kp["bbox_size"] * tx
        center_aug[1] += kp["bbox_size"] * ty

        kp3d_patch = _keypoint_3d_processing(
            kp["keypoints_3d"].astype(np.float32),
            IDENTITY_KP_PERMUTATION, rot_aug, do_flip,
        )
        img_patch_bgr, trans = _generate_image_patch_cv2(
            frame_bgr,
            float(center_aug[0]), float(center_aug[1]),
            kp["bbox_size"], kp["bbox_size"],
            self.patch_size, self.patch_size,
            do_flip, scale_aug, rot_aug,
        )
        mano_params_patch, has_mano_params_patch = _mano_param_processing(
            labels["mano_params"], labels["has_mano_params"], rot_aug, do_flip,
        )
        kp2d_patch = kp["keypoints_2d"].astype(np.float32).copy()
        if do_flip:
            kp2d_patch = _fliplr_keypoints(
                kp2d_patch, video_w, IDENTITY_KP_PERMUTATION
            )
        for j in range(len(kp2d_patch)):
            kp2d_patch[j, 0:2] = _trans_point2d(kp2d_patch[j, 0:2], trans)
        kp2d_patch[:, :-1] = kp2d_patch[:, :-1] / self.patch_size - 0.5

        img_patch = self._normalize_image(img_patch_bgr, color_scale)
        cam_info = {
            "video_w": video_w, "video_h": video_h,
            "K": K, "dist": dist, "cam_crop_params": cam_crop_params,
        }
        sample = self._build_sample(
            img_patch=img_patch,
            kp2d_patch=kp2d_patch,
            kp3d_patch=kp3d_patch,
            mano_params_patch=mano_params_patch,
            has_mano_params_patch=has_mano_params_patch,
            labels=labels,
            kp=kp,
            cam_info=cam_info,
            ep_idx=ep_idx,
            frame_idx=frame_idx,
            video_path=video_path,
            dataset_name="egoemg_wilor",
            frame_bgr=frame_bgr if self.return_frame_bgr else None,
        )
        if get_worker_info() is None:
            self.reset_runtime_caches()
        return sample

    def _getitem_per_episode_crops(self, idx: int) -> dict[str, Any] | None:
        """Read a crop from per-episode LMDB and reconstruct labels from memmaps."""
        ep_idx = int(self._sample_episode_idx[idx])
        frame_idx = int(self._sample_frame_idx[idx])
        hand_idx = int(self._sample_hand[idx])

        labels = self._read_memmap_labels(frame_idx, ep_idx, hand_idx)
        is_left = labels["is_left"]

        # Read pre-cropped image from LMDB
        lmdb_key = f"{labels['video_frame_idx']:08d}_{'L' if is_left else 'R'}"
        txn = self._get_pec_txn(ep_idx)
        if txn is None:
            return None
        jpeg_bytes = txn.get(lmdb_key.encode("ascii"))
        if jpeg_bytes is None:
            return None
        img_patch_bgr = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR,
        )
        if img_patch_bgr is None:
            return None

        video_w = self._pec_video_w
        video_h = self._pec_video_h
        K = self._pec_K
        dist = self._pec_dist
        intrinsics_info = self._pec_intrinsics_info
        cam_crop_params = _extract_crop_params(intrinsics_info, video_w, video_h)

        kp = self._project_keypoints_and_bbox(
            labels, K, dist, intrinsics_info, video_w, video_h
        )

        # Transform keypoints from raw video coords to the pre-cropped patch
        _, trans = _generate_image_patch_cv2(
            np.zeros((video_h, video_w, 3), dtype=np.uint8),
            float(kp["center"][0]), float(kp["center"][1]),
            kp["bbox_size"], kp["bbox_size"],
            self.patch_size, self.patch_size,
            do_flip=False, scale=1.0, rot=0.0,
        )
        kp2d_patch = kp["keypoints_2d"].astype(np.float32).copy()
        for j in range(len(kp2d_patch)):
            kp2d_patch[j, 0:2] = _trans_point2d(kp2d_patch[j, 0:2], trans)

        # Augmentation on the already-cropped patch
        if self.do_augment:
            scale_aug, rot_aug, do_flip, color_scale, tx, ty = (
                _do_augmentation(self.aug_config)
            )
            center_aug = np.array([
                self.patch_size / 2.0 + self.patch_size * tx,
                self.patch_size / 2.0 + self.patch_size * ty,
            ], dtype=np.float32)
            kp3d_patch = _keypoint_3d_processing(
                kp["keypoints_3d"].astype(np.float32),
                IDENTITY_KP_PERMUTATION, rot_aug, do_flip,
            )
            mano_params_patch, has_mano_params_patch = _mano_param_processing(
                labels["mano_params"], labels["has_mano_params"],
                rot_aug, do_flip,
            )
            img_patch_bgr, trans_aug = _generate_image_patch_cv2(
                img_patch_bgr,
                float(center_aug[0]), float(center_aug[1]),
                self.patch_size, self.patch_size,
                self.patch_size, self.patch_size,
                do_flip, scale_aug, rot_aug,
            )
            kp2d_aug = kp2d_patch.copy()
            if do_flip:
                kp2d_aug = _fliplr_keypoints(
                    kp2d_aug, self.patch_size, IDENTITY_KP_PERMUTATION,
                )
            for j in range(len(kp2d_aug)):
                kp2d_aug[j, 0:2] = _trans_point2d(kp2d_aug[j, 0:2], trans_aug)
            kp2d_patch = kp2d_aug
        else:
            color_scale = [1.0, 1.0, 1.0]
            kp3d_patch = kp["keypoints_3d"].astype(np.float32).copy()
            mano_params_patch = {
                k: v.copy() for k, v in labels["mano_params"].items()
            }
            has_mano_params_patch = labels["has_mano_params"]

        kp2d_patch[:, :-1] = kp2d_patch[:, :-1] / self.patch_size - 0.5
        img_patch = self._normalize_image(img_patch_bgr, color_scale)
        video_path = self._resolve_video_path(
            self._episode_head_video_path[ep_idx]
        )
        cam_info = {
            "video_w": video_w, "video_h": video_h,
            "K": K, "dist": dist, "cam_crop_params": cam_crop_params,
        }
        sample = self._build_sample(
            img_patch=img_patch,
            kp2d_patch=kp2d_patch,
            kp3d_patch=kp3d_patch,
            mano_params_patch=mano_params_patch,
            has_mano_params_patch=has_mano_params_patch,
            labels=labels,
            kp=kp,
            cam_info=cam_info,
            ep_idx=ep_idx,
            frame_idx=frame_idx,
            video_path=video_path,
            dataset_name="egoemg_wilor_per_episode",
        )
        if get_worker_info() is None:
            self.reset_runtime_caches()
        return sample
