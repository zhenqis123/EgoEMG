#!/usr/bin/env python3
"""Extract synchronized modality signals for the website hero showcase.

Consumes the ``<video>.frames.json`` sidecar written by
``visualize_dataset.py --dump-frames-map`` and packs EMG envelopes, wristband
IMU, and generated joint angles onto the *output video timeline* (uniform
grids), so a canvas panel can render them locked to ``video.currentTime``.

Outputs (into --out-dir):
  hero_signals.bin   concatenated float32 arrays (layout in the json)
  hero_signals.json  series descriptors + normalization ranges

Usage:
  python scripts/viz/extract_modality_showcase.py \
      --memmap-dir data/EgoEMG_full_memmap \
      --frames-json /tmp/hero/episode_000020_hero.frames.json \
      --out-dir <site>/assets/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EMG_RATE = 160.0     # Hz envelope output
IMU_RATE = 120.0     # Hz
JOINT_RATE = 30.0    # Hz
EMG_SMOOTH_MS = 20.0


def _memmap(memmap_dir: Path, manifest: dict, name: str) -> np.ndarray:
    spec = manifest["fields"][name]
    return np.memmap(
        memmap_dir / spec["filename"], dtype=spec["dtype"], mode="r",
        shape=tuple(spec["shape"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--frames-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--basename", default="hero_signals")
    args = ap.parse_args()

    frames = json.loads(args.frames_json.read_text())
    manifest = json.loads((args.memmap_dir / "manifest.json").read_text())
    rows = np.array([f["memmap_row"] for f in frames["frames"]], dtype=np.int64)
    out_fps = float(frames["out_fps"])
    duration = len(rows) / out_fps
    lo, hi = int(rows[0]), int(rows[-1])

    ts_us = _memmap(args.memmap_dir, manifest, "timestamp_us")
    # Real time of each output frame (anchors), then a uniform output grid.
    anchor_t = (ts_us[lo:hi + 1] - ts_us[lo]).astype(np.float64) / 1e6
    row_time = np.interp(np.arange(lo, hi + 1),
                         rows, np.arange(len(rows)) / out_fps)
    # row_time maps memmap row -> output time (piecewise linear on anchors).
    grid = lambda rate: np.arange(0, duration, 1.0 / rate)

    def to_out_time(t_real: np.ndarray) -> np.ndarray:
        return np.interp(t_real, anchor_t, row_time)

    series: dict[str, np.ndarray] = {}
    meta: dict = {
        "duration": float(duration),
        "out_fps": out_fps,
        "episode_id": frames["episode_id"],
        "series": {},
    }

    # ── EMG envelopes ──────────────────────────────────────────────────────
    win = max(1, int(2e3 * EMG_SMOOTH_MS / 1e3))  # rows in the smooth window
    for hand in ("left", "right"):
        sig = np.asarray(_memmap(
            args.memmap_dir, manifest, f"emg_{hand}_filtered_paper")[lo:hi + 1])
        env = np.abs(sig)
        kernel = np.ones(win) / win
        env = np.stack([
            np.convolve(env[:, c], kernel, mode="same") for c in range(env.shape[1])], axis=1)
        t_real = np.arange(lo, hi + 1) / 2e3 * 0 + anchor_t  # anchor_t IS row time
        t_out = to_out_time(anchor_t)
        g = grid(EMG_RATE)
        out = np.empty((len(g), env.shape[1]), dtype=np.float32)
        for c in range(env.shape[1]):
            out[:, c] = np.interp(g, t_out, env[:, c])
        vmax = float(np.percentile(out, 99.5)) or 1.0
        out = np.clip(out / vmax, 0.0, 1.0).astype(np.float32)
        series[f"emg_{hand}"] = out
        meta["series"][f"emg_{hand}"] = {
            "rate": EMG_RATE, "channels": int(out.shape[1]),
            "unit": "envelope (norm.)", "desc": f"{hand} wrist sEMG 2 kHz x 8ch",
        }

    # ── Wristband IMU ──────────────────────────────────────────────────────
    for hand in ("left", "right"):
        sig = np.asarray(_memmap(
            args.memmap_dir, manifest, f"imu_band_{hand}")[lo:hi + 1])
        g = grid(IMU_RATE)
        out = np.stack([
            np.interp(g, anchor_t, sig[:, c]) for c in range(6)], axis=1
        ).astype(np.float32)
        series[f"imu_{hand}"] = out
        meta["series"][f"imu_{hand}"] = {
            "rate": IMU_RATE, "channels": 6,
            "labels": ["ax", "ay", "az", "gx", "gy", "gz"],
            "range": [float(out.min()), float(out.max())],
            "desc": f"{hand} wrist IMU 120 Hz (accel + gyro)",
        }

    # ── Joint angles ───────────────────────────────────────────────────────
    for hand in ("left", "right"):
        sig = np.asarray(_memmap(
            args.memmap_dir, manifest,
            f"generated_joint_angles_{hand}")[lo:hi + 1])
        g = grid(JOINT_RATE)
        out = np.stack([
            np.interp(g, anchor_t, sig[:, c]) for c in range(sig.shape[1])], axis=1
        ).astype(np.float32)
        series[f"joints_{hand}"] = out
        meta["series"][f"joints_{hand}"] = {
            "rate": JOINT_RATE, "channels": int(out.shape[1]),
            "range": [float(out.min()), float(out.max())],
            "desc": f"{hand} hand 20-DoF joint angles (deg)",
        }

    # ── Pack ───────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blob = b"".join(s.astype("<f4").tobytes() for s in series.values())
    (args.out_dir / f"{args.basename}.bin").write_bytes(blob)
    offset = 0
    for key, s in series.items():
        meta["series"][key]["offset"] = offset
        meta["series"][key]["samples"] = int(s.shape[0])
        offset += s.size
    (args.out_dir / f"{args.basename}.json").write_text(json.dumps(meta, indent=1))
    total_kb = len(blob) / 1024
    print(f"packed {len(series)} series, {total_kb:.0f} KiB, "
          f"duration {duration:.2f}s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
