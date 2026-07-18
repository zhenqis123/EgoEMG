#!/usr/bin/env python3
"""Evaluate local small streaming inference on the Incre split from raw EMG."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from emg2pose.realtime_local.pipeline import LocalSmallStreamer


EXCLUDED_DEFAULT = {
    "data_20260526_172725",
    "data_20260526_230859",
    "data_20260527_124150",
    "sess_20260530_102930",
}


@dataclass
class RunningMetric:
    abs_sum: float = 0.0
    count: int = 0
    pred_count: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        err = np.abs(pred[:20] - target[:20])
        self.abs_sum += float(err.sum())
        self.count += int(err.size)
        self.pred_count += 1

    @property
    def mae(self) -> float:
        return self.abs_sum / self.count if self.count else float("nan")


def _decode(v) -> str:
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="replace").rstrip("\x00")
    return str(v)


def _open_field(root: Path, manifest: dict, name: str) -> np.memmap:
    spec = manifest["fields"][name]
    return np.memmap(
        root / spec["filename"],
        dtype=np.dtype(spec["dtype"]),
        mode="r",
        shape=tuple(spec["shape"]),
    )


def _split_ids(root: Path) -> dict[str, int]:
    meta = np.load(root / "metadata.npz", allow_pickle=False)
    splits = [_decode(x) for x in meta["splits_split"]]
    return {name: i for i, name in enumerate(splits)}


def _episode_end_exclusive(starts: np.ndarray, ends: np.ndarray, lengths: np.ndarray, total: int, ep_idx: int) -> int:
    start = int(starts[ep_idx])
    end = int(ends[ep_idx])
    length = int(lengths[ep_idx])
    if end - start == length:
        return min(end, total)
    if end - start + 1 == length:
        return min(end + 1, total)
    # Fall back to the next episode start when metadata conventions disagree.
    if ep_idx + 1 < len(starts):
        return min(int(starts[ep_idx + 1]), total)
    return min(total, end)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-dir", type=Path, default=Path("data/EgoEMG_incre/data_right_merged"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/tmp/egoemg-incre-small-8ch-runtime-21-12-36.pt"),
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--stride-samples", type=int, default=200)
    parser.add_argument("--window-length", type=int, default=12000)
    parser.add_argument("--output-delay-s", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-samples", type=int, default=200)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--max-preds-per-episode", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--include-excluded", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.memmap_dir
    with open(root / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    total = int(manifest["total_rows"])
    raw = _open_field(root, manifest, "emg_right_raw")
    angles = _open_field(root, manifest, "generated_joint_angles_right")
    valid = _open_field(root, manifest, "generated_label_valid")
    frame_split = _open_field(root, manifest, "frame_split_id")

    meta = np.load(root / "metadata.npz", allow_pickle=False)
    ep_ids = [_decode(x) for x in meta["episode_id"]]
    starts = meta["episode_start_idx"].astype(np.int64)
    ends = meta["episode_end_idx"].astype(np.int64)
    lengths = meta["episode_length"].astype(np.int64)
    split_id = _split_ids(root)[args.split]
    excluded = set() if args.include_excluded else EXCLUDED_DEFAULT

    overall = RunningMetric()
    by_episode: dict[str, RunningMetric] = defaultdict(RunningMetric)
    delay_samples = int(round(args.output_delay_s * 2000.0))
    chunk_samples = min(int(args.chunk_samples), int(args.stride_samples))
    if chunk_samples <= 0:
        raise ValueError("chunk-samples must be positive")

    print(
        f"checkpoint={args.checkpoint} split={args.split} stride={args.stride_samples} "
        f"delay_samples={delay_samples} chunk={chunk_samples}"
    )
    for ep_idx, ep_id in enumerate(ep_ids):
        if ep_id in excluded:
            print(f"skip excluded {ep_id}")
            continue
        ep_start = int(starts[ep_idx])
        ep_end = _episode_end_exclusive(starts, ends, lengths, total, ep_idx)
        ep_splits = np.asarray(frame_split[ep_start:ep_end], dtype=np.int32)
        if not np.any(ep_splits == split_id):
            print(f"skip no split={args.split}: {ep_id}")
            continue

        streamer = LocalSmallStreamer(
            checkpoint_path=args.checkpoint,
            stride_samples=args.stride_samples,
            device=args.device,
            input_scale=args.input_scale,
            output_delay_s=args.output_delay_s,
        )
        ep_metric = by_episode[ep_id]
        before = ep_metric.pred_count
        for local_s in range(0, ep_end - ep_start, chunk_samples):
            global_s = ep_start + local_s
            global_e = min(ep_end, global_s + chunk_samples)
            samples = np.asarray(raw[global_s:global_e], dtype=np.float32)
            preds = streamer.push_samples(samples, timestamp=0.0)
            for pred in preds:
                gt_idx = global_s + (global_e - global_s) - 1 - delay_samples
                if gt_idx < ep_start or gt_idx >= ep_end:
                    continue
                if int(frame_split[gt_idx]) != split_id:
                    continue
                if not bool(valid[gt_idx, 1]):
                    continue
                target = np.asarray(angles[gt_idx], dtype=np.float32)
                if not np.isfinite(target).all() or not np.isfinite(pred.angles[:20]).all():
                    continue
                overall.update(pred.angles, target)
                ep_metric.update(pred.angles, target)
                if args.max_preds_per_episode and ep_metric.pred_count >= args.max_preds_per_episode:
                    break
            if args.max_preds_per_episode and ep_metric.pred_count >= args.max_preds_per_episode:
                break
        print(
            f"{ep_id}: preds={ep_metric.pred_count - before} "
            f"mae={ep_metric.mae:.6f}"
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "memmap_dir": str(root),
        "split": args.split,
        "stride_samples": args.stride_samples,
        "output_delay_s": args.output_delay_s,
        "input_scale": args.input_scale,
        "overall": {
            "mae": overall.mae,
            "predictions": overall.pred_count,
            "values": overall.count,
        },
        "episodes": {
            ep: {"mae": metric.mae, "predictions": metric.pred_count, "values": metric.count}
            for ep, metric in by_episode.items()
        },
    }
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
