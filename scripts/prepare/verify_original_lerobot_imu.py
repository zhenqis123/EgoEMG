"""Verify IMU completeness/layout of the ORIGINAL EgoEMG LeRobot data.

Run this on the machine that hosts the original captures (Windows paths are
fine). It answers three questions before we touch the unified memmap:

  1. Does the original LeRobot parquet carry an IMU column at all
     (e.g. ``observation.imus``), and for how many episodes / rows?
  2. What is the native channel layout of that column --
     ``[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`` or gyro-first?
     The unified memmap's EgoEMG rows look gyro-first (gravity magnitude
     ~9.3 lives in channels 3-5, channel 0 is dead-zero), but the original
     schema is authoritative.
  3. Do the raw capture logs contain richer IMU streams (more channels,
     a live gyro axis, higher coverage) than the parquet column?

Usage (on the original-data machine)::

    python verify_original_lerobot_imu.py \
        --root "D:/develop/code/robot-data-collector/training_dataset_lerobot_full" \
        --captures "D:/develop/code/robot-data-collector/my_project/resources/logs/captures" \
        --max-episodes 8 --json-out imu_verify_report.json

Requires ``pyarrow`` (preferred) or ``pandas+pyarrow`` to read parquet.
Paste the printed report (or the JSON file) back into the analysis session.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GRAVITY = 9.80665


def _find_parquet_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*.parquet") if p.name.startswith("episode_"))
    return files or sorted(root.rglob("*.parquet"))


def _read_column(path: Path, column: str):
    """Read one parquet column as a 2-D float64 array (or None if not vector)."""
    import re

    import numpy as np
    import pyarrow.parquet as pq

    arr = pq.read_table(path, columns=[column]).column(0).combine_chunks()
    t = str(arr.type)
    if "fixed_size_list" in t:
        m = re.search(r"\[(\d+)\]$", t)
        width = int(m.group(1)) if m else 0
        flat = np.asarray(arr.flatten()).astype(np.float64)
        if width and flat.size % width == 0:
            return flat.reshape(-1, width)
        return None
    if "list" in t:  # variable-length lists
        try:
            return np.stack([np.asarray(v, dtype=np.float64) for v in arr.to_pylist()])
        except (ValueError, TypeError):
            return None
    values = np.asarray(arr).astype(np.float64)
    return values if values.ndim == 2 else None


def analyze_imu_column(path: Path, column: str) -> dict:
    import numpy as np

    data = _read_column(path, column)
    if data is None or data.ndim != 2:
        return {"episode": path.name, "column": column, "error": "not a fixed-width vector column"}
    n, width = data.shape
    rep: dict = {
        "episode": path.name,
        "column": column,
        "rows": int(n),
        "width": int(width),
        "nonzero_rows_pct": float((np.abs(data).sum(axis=1) > 0).mean() * 100),
    }
    if width < 6:
        rep["error"] = f"width {width} < 6, unexpected"
        return rep
    data = data[:, :6]
    stats = []
    for c in range(6):
        ch = data[:, c]
        stats.append(
            {
                "mean": float(ch.mean()),
                "std": float(ch.std()),
                "min": float(ch.min()),
                "max": float(ch.max()),
            }
        )
    rep["channels"] = stats
    rep["dead_channels"] = [c for c in range(6) if stats[c]["std"] == 0.0]
    for name, sl in (("first3", slice(0, 3)), ("last3", slice(3, 6))):
        mag = np.linalg.norm(data[:, sl], axis=1)
        rep[f"|{name}|_p50"] = float(np.percentile(mag, 50))
    first, last = rep["|first3|_p50"], rep["|last3|_p50"]
    d_first, d_last = abs(first - GRAVITY), abs(last - GRAVITY)
    if min(d_first, d_last) <= 2.0:
        rep["gravity_side"] = "first3" if d_first <= d_last else "last3"
        rep["layout_guess"] = (
            "[acc,gyro] (acc first)" if rep["gravity_side"] == "first3" else "[gyro,acc] (gyro first)"
        )
    else:
        rep["gravity_side"] = None
        rep["layout_guess"] = "no gravity-like half -- investigate"
    return rep


def sample_episodes(files: list[Path], max_episodes: int) -> list[Path]:
    if max_episodes <= 0 or max_episodes >= len(files):
        return files
    step = max(1, len(files) // max_episodes)
    picked = files[::step][:max_episodes]
    if files[-1] not in picked:
        picked.append(files[-1])
    return picked


def scan_captures(root: Path, limit: int = 40) -> list[dict]:
    hits = []
    for p in root.rglob("*"):
        if p.is_file() and "imu" in p.name.lower():
            hits.append({"path": str(p), "size_bytes": p.stat().st_size})
            if len(hits) >= limit:
                break
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path, help="LeRobot dataset root (training_dataset_lerobot_full)")
    ap.add_argument("--captures", type=Path, default=None, help="robot-data-collector captures logs root")
    ap.add_argument("--max-episodes", type=int, default=8, help="episodes to sample (0 = all)")
    ap.add_argument("--json-out", type=Path, default=None, help="write the full report as JSON")
    args = ap.parse_args()

    report: dict = {"root": str(args.root), "episodes": [], "captures_imu_files": []}

    files = _find_parquet_files(args.root)
    print(f"parquet episodes found: {len(files)}")
    if not files:
        print("No parquet files -- is this the right dataset root?")
    else:
        import pyarrow.parquet as pq

        cols = pq.read_schema(files[0]).names
        imu_cols = [c for c in cols if "imu" in c.lower()]
        print(f"columns of {files[0].name}: {cols}")
        print(f"imu-like columns: {imu_cols}")
        report["columns"] = cols
        report["imu_columns"] = imu_cols
        if not imu_cols:
            print("!! No IMU column in parquet schema -- backfill must use captures logs (Track B).")
        for f in sample_episodes(files, args.max_episodes):
            for col in imu_cols:
                try:
                    rep = analyze_imu_column(f, col)
                except Exception as exc:  # keep scanning remaining episodes
                    rep = {"episode": f.name, "column": col, "error": f"{type(exc).__name__}: {exc}"}
                report["episodes"].append(rep)
                if "error" in rep:
                    print(f"{rep['episode']} [{col}]: ERROR {rep['error']}")
                    continue
                print(
                    f"{rep['episode']} [{col}]: rows={rep['rows']} width={rep['width']} "
                    f"nonzero={rep['nonzero_rows_pct']:.1f}% |first3|p50={rep['|first3|_p50']:.2f} "
                    f"|last3|p50={rep['|last3|_p50']:.2f} -> {rep['layout_guess']} "
                    f"dead_ch={rep['dead_channels']}"
                )

    if args.captures:
        report["captures_root"] = str(args.captures)
        report["captures_imu_files"] = scan_captures(args.captures)
        print(f"\ncapture files with 'imu' in name: {len(report['captures_imu_files'])}")
        for h in report["captures_imu_files"][:10]:
            print(f"  {h['size_bytes']:>12,} B  {h['path']}")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
