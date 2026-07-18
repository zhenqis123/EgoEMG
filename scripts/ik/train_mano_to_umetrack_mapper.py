#!/usr/bin/env python3
"""Train a compact MANO-theta to UmeTrack-angle mapper.

The mapper is intended to replace per-frame IK in realtime WiLoR teacher
pipelines:

    MANO pose[:, 3:48] -> generated_joint_angles[:, :20]

It reads EgoEMG-style memmaps directly and exports a small runtime .pt that
contains model weights plus input/output normalization statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


HAND_TO_VALID_COL = {"left": 0, "right": 1}


class ManoToUmeTrackMapper(nn.Module):
    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(45, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 20),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Segment:
    root: Path
    hand: str
    start: int
    end: int
    split: str
    episode_name: str
    pose: np.memmap | np.ndarray
    angles: np.memmap | np.ndarray
    valid: np.memmap | np.ndarray | None

    @property
    def length(self) -> int:
        return self.end - self.start

    def preload(self) -> None:
        """Pre-load small segments into RAM for faster random access."""
        if isinstance(self.pose, np.memmap):
            self.pose = np.asarray(self.pose[self.start:self.end], dtype=np.float32)
        if isinstance(self.angles, np.memmap):
            self.angles = np.asarray(self.angles[self.start:self.end], dtype=np.float32)
        if self.valid is not None and isinstance(self.valid, np.memmap):
            self.valid = np.asarray(self.valid[self.start:self.end], dtype=bool)
        self.start = 0
        self.end = len(self.pose)


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _open_memmap(root: Path, manifest: dict, field: str) -> np.memmap | None:
    info = manifest.get("fields", {}).get(field)
    if info is None:
        return None
    return np.memmap(
        root / info["filename"],
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def _episode_ranges(root: Path, manifest: dict) -> list[tuple[int, int, str]]:
    meta_path = root / "metadata.npz"
    total_rows = int(manifest["total_rows"])
    if not meta_path.exists():
        return [(0, total_rows, root.name)]
    meta = np.load(meta_path, allow_pickle=True)
    if "episode_start_idx" not in meta.files or "episode_end_idx" not in meta.files:
        return [(0, total_rows, root.name)]
    starts = meta["episode_start_idx"].astype(np.int64)
    ends = meta["episode_end_idx"].astype(np.int64)
    names = meta["episode_id"] if "episode_id" in meta.files else np.arange(len(starts))
    ranges: list[tuple[int, int, str]] = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        # Some generated incre manifests store inclusive end indices.
        lengths = (
            meta["episode_length"].astype(np.int64)
            if "episode_length" in meta.files
            else np.full(len(starts), -1, dtype=np.int64)
        )
        if end_i < total_rows and (end_i - start_i + 1) == int(lengths[i]):
            end_i += 1
        ranges.append((start_i, min(end_i, total_rows), _decode(names[i])))
    return ranges


def discover_segments(
    roots: Iterable[Path],
    hands: Iterable[str],
    val_fraction: float,
    max_episode_frames: int | None,
) -> list[Segment]:
    segments: list[Segment] = []
    for root in roots:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            print(f"skip missing manifest: {root}")
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)

        valid = _open_memmap(root, manifest, "generated_label_valid")
        for hand in hands:
            pose = _open_memmap(root, manifest, f"generated_mano_{hand}_pose")
            angles = _open_memmap(root, manifest, f"generated_joint_angles_{hand}")
            if pose is None or angles is None:
                print(f"skip {root} hand={hand}: missing paired MANO/angles")
                continue
            if pose.shape[0] != angles.shape[0]:
                print(f"skip {root} hand={hand}: pose/angle length mismatch")
                continue

            ranges = _episode_ranges(root, manifest)
            n_episodes = len(ranges)
            if n_episodes == 1:
                start, end, name = ranges[0]
                val_len = int(round((end - start) * val_fraction))
                val_len = min(max(val_len, 1), max(end - start - 1, 1))
                train_end = max(start + 1, end - val_len)
                ranges = [
                    (start, train_end, f"{name}:train"),
                    (train_end, end, f"{name}:val"),
                ]
                n_episodes = len(ranges)
                val_episode_idx = {1}
            else:
                n_val = max(1, int(round(n_episodes * val_fraction)))
                val_episode_idx = set(range(max(0, n_episodes - n_val), n_episodes))
            for ep_idx, (start, end, name) in enumerate(ranges):
                if max_episode_frames is not None and end - start > max_episode_frames:
                    end = start + max_episode_frames
                split = "val" if ep_idx in val_episode_idx else "train"
                if end > start:
                    segments.append(
                        Segment(
                            root=root,
                            hand=hand,
                            start=start,
                            end=end,
                            split=split,
                            episode_name=name,
                            pose=pose,
                            angles=angles,
                            valid=valid,
                        )
                    )
    return segments


def _sample_indices(
    rng: np.random.Generator,
    segments: list[Segment],
    n: int,
    sampling: str = "frame",
) -> tuple[list[Segment], list[np.ndarray]]:
    if sampling == "frame":
        weights = np.asarray([s.length for s in segments], dtype=np.float64)
    elif sampling == "segment":
        weights = np.ones(len(segments), dtype=np.float64)
    elif sampling == "root":
        roots = sorted({str(s.root) for s in segments})
        root_prob = {root: 1.0 / len(roots) for root in roots}
        root_lengths = {
            root: sum(s.length for s in segments if str(s.root) == root)
            for root in roots
        }
        weights = np.asarray(
            [
                root_prob[str(s.root)] * s.length / max(root_lengths[str(s.root)], 1)
                for s in segments
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")
    weights = weights / weights.sum()
    counts = rng.multinomial(n, weights)
    out_segments: list[Segment] = []
    out_indices: list[np.ndarray] = []
    for segment, count in zip(segments, counts):
        if count <= 0:
            continue
        idx = rng.integers(segment.start, segment.end, size=int(count), endpoint=False)
        idx.sort()
        out_segments.append(segment)
        out_indices.append(idx.astype(np.int64))
    return out_segments, out_indices


def _load_samples(
    segments: list[Segment],
    indices: list[np.ndarray],
    angle_abs_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for segment, idx in zip(segments, indices):
        # Bulk read contiguous chunks where possible
        if idx.max() - idx.min() < 50000:
            chunk_pose = np.asarray(segment.pose[idx, 3:48], dtype=np.float32)
            chunk_angles = np.asarray(segment.angles[idx, :20], dtype=np.float32)
        else:
            # Sorted indices but sparse → read individual rows (still faster with preloaded arrays)
            chunk_pose = np.stack([np.asarray(segment.pose[i, 3:48], dtype=np.float32) for i in idx])
            chunk_angles = np.stack([np.asarray(segment.angles[i, :20], dtype=np.float32) for i in idx])
        mask = np.isfinite(chunk_pose).all(axis=1) & np.isfinite(chunk_angles).all(axis=1)
        mask &= np.abs(chunk_angles).max(axis=1) <= angle_abs_limit
        if segment.valid is not None:
            valid = np.asarray(segment.valid[idx])
            if valid.ndim == 2 and valid.shape[1] >= 2:
                mask &= valid[:, HAND_TO_VALID_COL[segment.hand]].astype(bool)
            else:
                mask &= valid.astype(bool)
        if mask.any():
            xs.append(chunk_pose[mask])
            ys.append(chunk_angles[mask])
    if not xs:
        return np.empty((0, 45), dtype=np.float32), np.empty((0, 20), dtype=np.float32)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def sample_batch(
    rng: np.random.Generator,
    segments: list[Segment],
    batch_size: int,
    angle_abs_limit: float,
    oversample: int = 2,
    sampling: str = "frame",
) -> tuple[np.ndarray, np.ndarray]:
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    attempts = 0
    while sum(x.shape[0] for x in xs_all) < batch_size and attempts < 8:
        attempts += 1
        segs, idxs = _sample_indices(
            rng,
            segments,
            batch_size * oversample,
            sampling=sampling,
        )
        x, y = _load_samples(segs, idxs, angle_abs_limit)
        if x.shape[0]:
            xs_all.append(x)
            ys_all.append(y)
    if not xs_all:
        raise RuntimeError("Could not sample a valid batch")
    x = np.concatenate(xs_all, axis=0)[:batch_size]
    y = np.concatenate(ys_all, axis=0)[:batch_size]
    return x, y


def compute_stats(
    rng: np.random.Generator,
    segments: list[Segment],
    n_samples: int,
    angle_abs_limit: float,
    sampling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = sample_batch(
        rng,
        segments,
        n_samples,
        angle_abs_limit=angle_abs_limit,
        oversample=3,
        sampling=sampling,
    )
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    x_std = np.maximum(x_std, 1e-6)
    y_std = np.maximum(y_std, 1e-6)
    return (
        x_mean.astype(np.float32),
        x_std.astype(np.float32),
        y_mean.astype(np.float32),
        y_std.astype(np.float32),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    segments: list[Segment],
    rng: np.random.Generator,
    device: torch.device,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    n_samples: int,
    batch_size: int,
    angle_abs_limit: float,
    sampling: str,
) -> dict[str, float]:
    model.eval()
    total_abs = 0.0
    total_abs_deg = 0.0
    total = 0
    total_loss = 0.0
    remaining = n_samples
    while remaining > 0:
        cur = min(batch_size, remaining)
        x_np, y_np = sample_batch(
            rng,
            segments,
            cur,
            angle_abs_limit,
            sampling=sampling,
        )
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        x_n = (x - x_mean) / x_std
        y_n = (y - y_mean) / y_std
        pred_n = model(x_n)
        loss = torch.nn.functional.smooth_l1_loss(pred_n, y_n)
        pred = pred_n * y_std + y_mean
        abs_err = (pred - y).abs()
        total_loss += float(loss.item()) * x.shape[0]
        total_abs += float(abs_err.sum().item())
        total_abs_deg += float(torch.rad2deg(abs_err).sum().item())
        total += int(abs_err.numel())
        remaining -= cur
    return {
        "loss": total_loss / max(n_samples, 1),
        "mae_rad": total_abs / max(total, 1),
        "mae_deg": total_abs_deg / max(total, 1),
    }


def default_roots() -> list[Path]:
    roots = [Path("data/EgoEMG_memmap")]
    roots.extend(sorted(Path("data").glob("sess_*/memmap")))
    roots.append(Path("data/EgoEMG_incre/data_right_merged"))
    return roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, action="append", default=None)
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("pretrained_models/mano_to_umetrack_mapper.pt"))
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stats-samples", type=int, default=200000)
    parser.add_argument("--val-samples", type=int, default=50000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--sampling",
        choices=["root", "frame", "segment"],
        default="root",
        help=(
            "Training sample weighting. root gives each data root equal mass, "
            "which keeps small incre sessions from being drowned by EgoEMG."
        ),
    )
    parser.add_argument("--max-episode-frames", type=int, default=None)
    parser.add_argument("--angle-abs-limit", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = args.data_root if args.data_root is not None else default_roots()
    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    segments = discover_segments(
        roots=roots,
        hands=hands,
        val_fraction=args.val_fraction,
        max_episode_frames=args.max_episode_frames,
    )
    train_segments = [s for s in segments if s.split == "train"]
    val_segments = [s for s in segments if s.split == "val"]
    if not train_segments or not val_segments:
        raise SystemExit(
            f"Need non-empty train and val segments, got train={len(train_segments)} "
            f"val={len(val_segments)}"
        )

    def _summarize(name: str, segs: list[Segment]) -> None:
        rows = sum(s.length for s in segs)
        print(f"{name}: {len(segs)} segments, {rows:,} candidate frames")
        root_rows: dict[str, int] = {}
        for s in segs:
            root_rows[str(s.root)] = root_rows.get(str(s.root), 0) + s.length
        for root, count in sorted(root_rows.items(), key=lambda item: item[0]):
            print(f"  root {count:>12,} frames  {root}")
        for s in segs[:8]:
            print(f"  {s.split:5s} {s.hand:5s} {s.root} {s.episode_name} [{s.start}, {s.end})")
        if len(segs) > 8:
            print(f"  ... {len(segs) - 8} more")

    _summarize("train", train_segments)
    _summarize("val", val_segments)

    # Preload small segments into RAM for faster random access.
    # Large datasets like EgoEMG stay as memmaps.
    PRELOAD_THRESHOLD = 500_000
    for seg in train_segments + val_segments:
        if seg.length <= PRELOAD_THRESHOLD:
            seg.preload()
    n_preloaded = sum(1 for s in train_segments + val_segments if isinstance(s.pose, np.ndarray))
    print(f"preloaded {n_preloaded}/{len(train_segments)+len(val_segments)} segments into RAM")

    rng = np.random.default_rng(args.seed)
    print(f"computing normalization stats from {args.stats_samples:,} samples")
    print(f"sampling mode: {args.sampling}")
    x_mean_np, x_std_np, y_mean_np, y_std_np = compute_stats(
        rng,
        train_segments,
        args.stats_samples,
        args.angle_abs_limit,
        sampling=args.sampling,
    )

    device = torch.device(args.device)
    model = ManoToUmeTrackMapper(hidden_dim=args.hidden_dim).to(device)
    x_mean = torch.from_numpy(x_mean_np).to(device)
    x_std = torch.from_numpy(x_std_np).to(device)
    y_mean = torch.from_numpy(y_mean_np).to(device)
    y_std = torch.from_numpy(y_std_np).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * args.steps_per_epoch, 1),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_mae = math.inf
    best_payload: dict | None = None
    train_rng = np.random.default_rng(args.seed + 1)
    val_rng = np.random.default_rng(args.seed + 2)

    print(f"training on {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses: list[float] = []
        for _ in range(args.steps_per_epoch):
            x_np, y_np = sample_batch(
                train_rng,
                train_segments,
                args.batch_size,
                args.angle_abs_limit,
                sampling=args.sampling,
            )
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            x_n = (x - x_mean) / x_std
            y_n = (y - y_mean) / y_std
            pred_n = model(x_n)
            loss = torch.nn.functional.smooth_l1_loss(pred_n, y_n)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        metrics = evaluate(
            model,
            val_segments,
            val_rng,
            device,
            x_mean,
            x_std,
            y_mean,
            y_std,
            n_samples=args.val_samples,
            batch_size=args.batch_size,
            angle_abs_limit=args.angle_abs_limit,
            sampling=args.sampling,
        )
        train_loss = float(np.mean(losses))
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.5f} val_loss={metrics['loss']:.5f} "
            f"val_mae={metrics['mae_rad']:.5f}rad/{metrics['mae_deg']:.2f}deg "
            f"lr={lr:.2e} time={elapsed:.1f}s"
        )

        payload = {
            "model": "ManoToUmeTrackMapper",
            "hidden_dim": args.hidden_dim,
            "input_dim": 45,
            "output_dim": 20,
            "input_semantics": "MANO axis-angle pose without global_orient, pose[:, 3:48]",
            "output_semantics": "UmeTrack generated_joint_angles first 20 finger angles",
            "state_dict": model.state_dict(),
            "x_mean": torch.from_numpy(x_mean_np),
            "x_std": torch.from_numpy(x_std_np),
            "y_mean": torch.from_numpy(y_mean_np),
            "y_std": torch.from_numpy(y_std_np),
            "epoch": epoch,
            "val_metrics": metrics,
            "train_loss": train_loss,
            "data_roots": [str(p) for p in roots],
            "hands": hands,
            "sampling": args.sampling,
        }
        torch.save(payload, args.output.with_suffix(".last.pt"))
        if metrics["mae_rad"] < best_mae:
            best_mae = metrics["mae_rad"]
            best_payload = payload
            torch.save(payload, args.output)
            print(f"  saved best -> {args.output}")

    if best_payload is None:
        raise RuntimeError("Training finished without a best checkpoint")
    print(
        f"done: best val_mae={best_payload['val_metrics']['mae_rad']:.5f}rad/"
        f"{best_payload['val_metrics']['mae_deg']:.2f}deg at epoch {best_payload['epoch']}"
    )


if __name__ == "__main__":
    main()
