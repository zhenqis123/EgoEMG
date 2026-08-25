#!/usr/bin/env python3
"""Precompute frozen ViT features into per-episode LMDBs mirroring the crops layout.

Reads pre-cropped JPEG patches from per-episode LMDBs, runs them through a frozen
ViT backbone, and writes 1280-dim float32 features to output LMDBs with the
identical key scheme::

    {frame_idx:08d}_{L/R}  ->  (1280,) float32 bytes

Uses PyTorch DataLoader with multiprocessing workers for parallel JPEG decode,
mirroring the training-time data pipeline to keep the GPU saturated.

Output structure::

    {output_dir}/
        episode_000000.lmdb/
        episode_000000.done
        ...

No manifest, no split directories, no .npy files. Resume is automatic via .done
markers — existing episodes are skipped.

Usage:
    python scripts/prepare/precompute_vit_features_to_lmdb.py \\
        --crops-dir data/EgoEMG_crops \\
        --output-dir data/EgoEMG_v2_vit_features_lmdb \\
        --pretrained-path ../WiLoR/pretrained_models/wilor_final.ckpt

    # Parallel across GPUs:
    CUDA_VISIBLE_DEVICES=0 python scripts/prepare/precompute_vit_features_to_lmdb.py \\
        ... --episode-start 0 --episode-end 10
    CUDA_VISIBLE_DEVICES=1 python scripts/prepare/precompute_vit_features_to_lmdb.py \\
        ... --episode-start 10 --episode-end 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

FEAT_DIM = 1280

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


def _decode_one(buf: bytes) -> np.ndarray | None:
    """Decode a single JPEG and return (3, 256, 256) float32 normalized image."""
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = np.transpose(img[:, :, ::-1], (2, 0, 1)).astype(np.float32)
    for c in range(3):
        img[c, :, :] = (img[c, :, :] - _MEAN[c]) / _STD[c]
    return img


class _VitFeatureDataset(Dataset):
    """Lightweight Dataset: maps index → (key, decoded normalised image).

    JPEG bytes are held by reference; DataLoader workers inherit them via fork
    (Linux default), so there is zero serialization cost.
    """

    def __init__(self, keys: list[bytes], jpeg_list: list[bytes]) -> None:
        self.keys = keys
        self.jpeg_list = jpeg_list

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[bytes, np.ndarray | None]:
        return self.keys[idx], _decode_one(self.jpeg_list[idx])


def _collate_fn(
    batch: list[tuple[bytes, np.ndarray | None]],
) -> tuple[list[bytes], torch.Tensor | None]:
    """Drop failed decodes, stack valid images into a single tensor."""
    valid_keys: list[bytes] = []
    valid_imgs: list[np.ndarray] = []
    for key, img in batch:
        if img is not None:
            valid_keys.append(key)
            valid_imgs.append(img)
    if not valid_imgs:
        return [], None
    tensor = torch.stack([torch.from_numpy(img) for img in valid_imgs])
    return valid_keys, tensor


def _build_backbone(pretrained_path: str, device: torch.device):
    from egoemg.models.vision_only_angle import VisionOnlyAngleModule

    module = VisionOnlyAngleModule(
        pretrained_path=pretrained_path,
        mano_model_path=None,
    )
    backbone = module.vision_backbone
    backbone.eval()
    backbone.to(device)
    del module
    return backbone


def _extract_features(
    tensor: torch.Tensor, backbone: torch.nn.Module,
) -> np.ndarray:
    """Run ViT backbone and return (B, FEAT_DIM) float32 features on CPU."""
    with torch.no_grad():
        out = backbone(tensor)

    if isinstance(out, tuple):
        feat = None
        for item in reversed(out):
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                feat = item
                break
        if feat is None:
            raise ValueError("backbone returned no 4D feature map")
    else:
        feat = out

    return feat.mean(dim=[-2, -1]).cpu().numpy().astype(np.float32)


def _process_episode(
    ep_lmdb_path: Path,
    out_lmdb_path: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> int:
    """Extract ViT features for all keys in a crops LMDB, write to output LMDB.

    Uses DataLoader with multiprocess workers — the same infrastructure
    that saturates the GPU during training.
    """
    if not ep_lmdb_path.exists():
        log.warning("Crops LMDB not found: %s", ep_lmdb_path)
        return 0

    # ── Read all keys and JPEG bytes ──────────────────────────────────────
    env_in = lmdb.open(str(ep_lmdb_path), readonly=True, lock=False, readahead=False)
    keys: list[bytes] = []
    jpeg_list: list[bytes] = []
    with env_in.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor.iternext(keys=True, values=True):
            keys.append(bytes(key))
            jpeg_list.append(bytes(value))
    env_in.close()

    if not keys:
        out_env = lmdb.open(str(out_lmdb_path), map_size=1024 * 1024)
        out_env.close()
        return 0

    n_total = len(keys)

    # ── Build DataLoader over the raw JPEG list ───────────────────────────
    dataset = _VitFeatureDataset(keys, jpeg_list)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=_collate_fn,
        drop_last=False,
    )

    total_batches = (n_total + batch_size - 1) // batch_size
    log_interval = max(1, total_batches // 10)

    # ── Open output LMDB ──────────────────────────────────────────────────
    # Conservative estimate: 20 KB per entry (covers 5.1 KB feature + key + B-tree overhead).
    map_size = max(n_total * 20_000, 100 * 1024 * 1024)
    env_out = lmdb.open(str(out_lmdb_path), map_size=map_size)
    txn = env_out.begin(write=True)
    total_written = 0
    commit_interval = 5000

    for batch_idx, (batch_keys, tensor) in enumerate(dataloader, start=1):
        if tensor is None:
            continue

        # Crop height 256 -> 192 (ViT positional embedding is 192x256).
        tensor = tensor[:, :, 32:-32, :].to(device, non_blocking=True)

        features = _extract_features(tensor, backbone)

        for key, feat in zip(batch_keys, features):
            txn.put(key, feat.tobytes())

        total_written += len(batch_keys)

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            pct = batch_idx * 100.0 / total_batches
            log.info("[%s] batch %d/%d (%.0f%%), %d features written",
                     ep_lmdb_path.stem, batch_idx, total_batches, pct, total_written)

        if total_written % commit_interval < len(batch_keys):
            txn.commit()
            txn = env_out.begin(write=True)

    txn.commit()
    env_out.sync()
    env_out.close()

    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Precompute ViT features into per-episode LMDBs"
    )
    parser.add_argument("--crops-dir", required=True, help="Per-episode crops LMDB directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-path", required=True, help="WiLoR checkpoint path")
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker processes for parallel decode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    crops_dir = Path(args.crops_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover episodes from .done files in crops dir ───────────────────
    done_files = sorted(crops_dir.glob("*.done"))
    if not done_files:
        log.error("No .done files found in %s", crops_dir)
        sys.exit(1)

    all_episodes = [f.stem for f in done_files]
    ep_start = args.episode_start if args.episode_start is not None else 0
    ep_end = (
        args.episode_end
        if args.episode_end is not None
        else len(all_episodes)
    )
    ep_start = max(0, min(ep_start, len(all_episodes)))
    ep_end = max(ep_start, min(ep_end, len(all_episodes)))

    log.info(
        "Episode range: [%d, %d) / %d total",
        ep_start, ep_end, len(all_episodes),
    )

    # ── Build frozen backbone once ────────────────────────────────────────
    log.info("Loading ViT backbone from %s ...", args.pretrained_path)
    backbone = _build_backbone(args.pretrained_path, device)
    log.info("Backbone ready on %s", args.device)

    # ── Process each episode ──────────────────────────────────────────────
    t_start = time.time()
    total_features = 0
    processed = 0
    skipped = 0

    for ep_idx in range(ep_start, ep_end):
        ep_name = all_episodes[ep_idx]
        done_path = output_dir / f"{ep_name}.done"
        if done_path.exists():
            skipped += 1
            continue

        ep_lmdb_path = crops_dir / f"{ep_name}.lmdb"
        out_lmdb_path = output_dir / f"{ep_name}.lmdb"

        if out_lmdb_path.exists():
            import shutil
            shutil.rmtree(out_lmdb_path)

        log.info("[%d/%d] %s ...", ep_idx + 1, ep_end, ep_name)
        n = _process_episode(
            ep_lmdb_path, out_lmdb_path, backbone, device,
            args.batch_size, args.num_workers,
        )
        done_path.write_text(str(n))
        total_features += n
        processed += 1
        log.info("[%d/%d] %s done: %d features", ep_idx + 1, ep_end, ep_name, n)

    elapsed = time.time() - t_start
    log.info(
        "Done: %d episodes processed, %d skipped, %d features in %.1f s",
        processed, skipped, total_features, elapsed,
    )


if __name__ == "__main__":
    main()
