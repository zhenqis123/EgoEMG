#!/usr/bin/env python3
"""Verify a prepared EgoEMG vision profile against the live dataset path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WILOR_PATH = Path(__file__).resolve().parents[2] / "WiLoR"
if str(WILOR_PATH) not in sys.path:
    sys.path.insert(0, str(WILOR_PATH))

from wilor.configs import get_config

from emg2pose.datasets.egoemg_vision_dataset import EgoEmgVisionDataset


TENSOR_FIELDS = ["global_orient", "hand_pose", "betas", "joint_angles"]
SCALAR_FLOAT_FIELDS = ["raw_right", "is_right", "right", "box_size", "label_valid_mask"]
SCALAR_INT_FIELDS = ["frame_index", "video_frame_index"]
STRING_FIELDS = ["video_path", "episode_id", "episode_subject", "bbox_source_name"]


def _ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    sigma1 = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1**2
    sigma2 = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2**2
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2
    return float(
        (
            ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2))
            / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        ).mean()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--allintra-root", type=Path, required=True)
    parser.add_argument("--vision-index-dir", type=Path, default=None)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--wilor-model-config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    split_dir = args.profile_dir / args.split
    with (split_dir / "split_manifest.json").open("r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    model_config_path = args.wilor_model_config or str(
        WILOR_PATH / "pretrained_models" / "model_config.yaml"
    )
    wilor_cfg = get_config(model_config_path, merge=True, update_cachedir=False)
    wilor_cfg.defrost()
    if "PRETRAINED_WEIGHTS" in wilor_cfg.MODEL.BACKBONE:
        wilor_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    wilor_cfg.freeze()

    patch_size = int(split_manifest.get("patch_size", wilor_cfg.MODEL.IMAGE_SIZE))
    mean = 255.0 * torch.tensor(wilor_cfg.MODEL.IMAGE_MEAN).numpy()
    std = 255.0 * torch.tensor(wilor_cfg.MODEL.IMAGE_STD).numpy()
    ds_kwargs = dict(
        memmap_dir=args.memmap_dir,
        video_root=args.video_root,
        allintra_root=args.allintra_root,
        vision_index_dir=args.vision_index_dir,
        calibration_path=args.calibration_path,
        target_hand=split_manifest.get("target_hand", "both"),
        patch_size=patch_size,
        mean=mean,
        std=std,
        aug_config=wilor_cfg.DATASETS.CONFIG,
        do_augment=False,
    )
    live_ds = EgoEmgVisionDataset(
        **ds_kwargs,
        allowed_splits=split_manifest["allowed_splits"],
        stride=int(split_manifest["stride"]),
    )
    prepared_ds = EgoEmgVisionDataset(
        **ds_kwargs,
        prepared_crops_dir=split_dir,
    )

    n = min(int(args.num_samples), len(live_ds), len(prepared_ds))
    print(
        f"Comparing {n} samples "
        f"(live={len(live_ds)}, prepared={len(prepared_ds)}, split={args.split})"
    )
    n_img_fail = 0
    n_tensor_fail = 0
    n_string_fail = 0
    ssims = []
    for i in range(n):
        live = live_ds[i]
        prepared = prepared_ds[i]
        ssim = _ssim(live["img"], prepared["img"])
        ssims.append(ssim)
        if ssim < 0.90:
            n_img_fail += 1
            rmse = float(np.sqrt(np.mean((live["img"] - prepared["img"]) ** 2)))
            print(f"[{i}] img fail rmse={rmse:.2f} ssim={ssim:.4f}")

        for field in TENSOR_FIELDS:
            if not np.allclose(live[field], prepared[field], atol=1e-4):
                n_tensor_fail += 1
                diff = float(np.abs(live[field] - prepared[field]).max())
                print(f"[{i}] {field} fail max_diff={diff:.6f}")

        if not np.allclose(live["keypoints_2d"], prepared["keypoints_2d"], atol=1e-4):
            n_tensor_fail += 1
            diff = float(np.abs(live["keypoints_2d"] - prepared["keypoints_2d"]).max())
            print(f"[{i}] keypoints_2d fail max_diff={diff:.6f}")
        if not np.allclose(
            live["keypoints_3d"][:, :3],
            prepared["keypoints_3d"][:, :3],
            atol=1e-4,
        ):
            n_tensor_fail += 1
            diff = float(
                np.abs(live["keypoints_3d"][:, :3] - prepared["keypoints_3d"][:, :3]).max()
            )
            print(f"[{i}] keypoints_3d fail max_diff={diff:.6f}")

        for field in SCALAR_FLOAT_FIELDS:
            if abs(float(live[field]) - float(prepared[field])) > 1e-4:
                n_tensor_fail += 1
                print(f"[{i}] {field} fail live={live[field]} prepared={prepared[field]}")
        for field in SCALAR_INT_FIELDS:
            if int(live[field]) != int(prepared[field]):
                n_tensor_fail += 1
                print(f"[{i}] {field} fail live={live[field]} prepared={prepared[field]}")
        for field in STRING_FIELDS:
            if str(live[field]) != str(prepared[field]):
                n_string_fail += 1
                print(f"[{i}] {field} fail live={live[field]!r} prepared={prepared[field]!r}")

    print(f"SSIM mean={np.mean(ssims):.4f} min={np.min(ssims):.4f}")
    print(
        f"image_failures={n_img_fail} tensor_failures={n_tensor_fail} "
        f"string_failures={n_string_fail}"
    )
    if n_img_fail or n_tensor_fail or n_string_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
