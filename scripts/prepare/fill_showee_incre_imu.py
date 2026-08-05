"""Fill the missing IMU modalities in the unified memmap.

The released memmap carries a single 6-dim ``imu`` field (EgoEMG +
ShowEE left wrist band).  The recordings also contain:

  * ShowEE right wrist band IMU  (``wavelet_right_wrist/wavelet.h5``)
  * ShowEE camera IMUs           (``showee_{head,left_wrist,right_wrist}/imu_*.json``,
                                  ~420 Hz, timestamped)
  * Incre wrist band IMU         (``WeiLiEMG_13_COM3/weili_imu.csv``,
                                  gyro-first, timestamped)

This script writes four new float32 (N, 6) fields
(``imu_right`` / ``imu_head`` / ``imu_wrist_left`` / ``imu_wrist_right``,
acceleration first, matching the existing ``imu`` layout) by nearest-
timestamp sampling onto the 2 kHz row grid.  Rows without a source are
zero-filled (no sensor data), consistent with the memmap convention.

Incre's IMU (right wrist band) goes into ``imu_right``; the three
``data_20260526/27`` Incre sessions have no local copy of their
``weili_imu.csv`` and stay zero.

Usage::

    python scripts/prepare/fill_showee_incre_imu.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_unified_memmap \
        --source-memmap-dir /data/xiziheng/EgoEMG_unified_memmap \
        --showee-root /mnt/nvme/xiziheng/showee/downloads \
        --showee-root /mnt/nvme/xiziheng \
        --incre-root /home/xiziheng/develop/emg2pose/data
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

FIELDS = ("imu_right", "imu_head", "imu_wrist_left", "imu_wrist_right")


def _clean(values) -> list[str]:
    out = []
    for v in values:
        s = v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
        s = s.strip("b'").strip('"')
        out.append(s)
    return out


def _find_dir(roots: list[Path], name: str) -> Path | None:
    for root in roots:
        p = root / name
        if p.is_dir():
            return p
    return None


def _nearest_indices(source_us: np.ndarray, target_us: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_us, target_us, side="left")
    right = np.clip(right, 0, len(source_us) - 1)
    left = np.clip(right - 1, 0, len(source_us) - 1)
    choose_left = np.abs(target_us - source_us[left]) <= np.abs(
        source_us[right] - target_us)
    return np.where(choose_left, left, right).astype(np.int64)


def _read_band_imu(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ts_us, imu) from a wavelet h5 (acc, gyro) or None."""
    import h5py
    with h5py.File(h5_path, "r") as h:
        if "imu" not in h:
            return None
        acc = h["imu/acc"][:].astype(np.float32)
        gyro = h["imu/gyro"][:].astype(np.float32)
        ts = h["imu/timestamp"][:].astype(np.int64)
    return ts, np.concatenate([acc, gyro], axis=1)


def _read_camera_imu(json_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(ts_us, imu) from a camera imu_*.json (ts, acc, gyro)."""
    text = json_path.read_text()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        # Some camera IMU logs were truncated mid-write (recording
        # interrupted).  The rows are one per line: keep every complete
        # row and drop the truncated tail.
        rows = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # truncated row; the rest of the file is unusable
    arr = np.asarray(rows, dtype=np.float64)
    ts = arr[:, 0].astype(np.int64)
    # json layout: [ts, ax, ay, az, gx, gy, gz] -> memmap [acc, gyro]
    imu = arr[:, 1:7].astype(np.float32)
    return ts, imu


def _read_weili_imu(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ts_us, imu) from weili_imu.csv (timestamp, gyro, acc)."""
    ts: list[int] = []
    gyro: list[tuple[float, float, float]] = []
    acc: list[tuple[float, float, float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(int(row["timestamp_us"]))
            gyro.append((float(row["gyro_x"]), float(row["gyro_y"]),
                         float(row["gyro_z"])))
            acc.append((float(row["acc_x"]), float(row["acc_y"]),
                        float(row["acc_z"])))
    acc = np.asarray(acc, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    # The WeiLi EMG band reports |acc| ~= 5.4 while static (vs 9.8 m/s^2
    # for the other sensors): scale so the gravity magnitude matches the
    # memmap's m/s^2 convention.
    acc = acc * (9.8 / 5.4)
    imu = np.concatenate([acc, gyro], axis=1)
    return np.asarray(ts, dtype=np.int64), imu


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--source-memmap-dir", type=Path, required=True,
                    help="Memmap holding the per-action (928-episode) "
                         "metadata backup (metadata.npz.orig928).")
    ap.add_argument("--showee-root", type=Path, nargs="+", required=True)
    ap.add_argument("--incre-root", type=Path, required=True,
                    help="Directory containing the Incre session folders "
                         "(data/sess_2026* with WeiLiEMG_13_COM3/weili_imu.csv).")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    target = np.load(args.memmap_dir / "metadata.npz", allow_pickle=False)
    with (args.memmap_dir / "manifest.json").open() as f:
        manifest = json.load(f)
    src_path = args.source_memmap_dir / "metadata.npz.orig928"
    if not src_path.is_file():
        src_path = args.source_memmap_dir / "metadata.npz"
    src = np.load(src_path, allow_pickle=False)
    print(f"source metadata: {src_path.name} ({len(src['episode_id'])} episodes)")

    n_rows = int(manifest["total_rows"])
    mmaps = {}
    for name in FIELDS:
        if name in manifest["fields"]:
            info = manifest["fields"][name]
            mmaps[name] = np.memmap(
                args.memmap_dir / info["filename"], mode="r+",
                dtype=info["dtype"], shape=tuple(info["shape"]))
        elif not args.dry_run:
            info = {"filename": f"{name}.dat", "dtype": "float32",
                    "shape": [n_rows, 6]}
            manifest["fields"][name] = info
            mm = np.memmap(args.memmap_dir / f"{name}.dat", mode="w+",
                           dtype="float32", shape=(n_rows, 6))
            mm[:] = 0.0
            mm.flush()
            mmaps[name] = mm
            print(f"created field {name}")

    ts_mm = np.memmap(args.memmap_dir / "timestamp_us.dat",
                      dtype=np.int64, mode="r", shape=(n_rows,))
    target_parquet = _clean(target["episode_source_parquet"])
    src_parquet = _clean(src["episode_source_parquet"])
    src_start = src["episode_start_idx"].astype(np.int64)
    src_end = src["episode_end_idx"].astype(np.int64)

    # ── ShowEE: band + camera IMUs ─────────────────────────────────────
    session_eps = [
        i for i, p in enumerate(target_parquet)
        if p and _find_dir(args.showee_root, p) is not None
    ]
    print(f"{len(session_eps)} ShowEE sessions")
    n_filled = {f: 0 for f in FIELDS}
    for ep_idx in session_eps:
        session = target_parquet[ep_idx]
        s = int(target["episode_start_idx"][ep_idx])
        e = int(target["episode_end_idx"][ep_idx])
        if e == s + int(target["episode_length"][ep_idx]) - 1:
            e += 1
        session_dir = _find_dir(args.showee_root, session)
        assert session_dir is not None
        actions = sorted(
            [i for i, p in enumerate(src_parquet) if p.startswith(session + "/")],
            key=lambda i: int(src_start[i]))
        for a in actions:
            a_name = src_parquet[a].split("/")[1]
            a_dir = session_dir / a_name
            if not a_dir.is_dir() and a_name == "thum" \
                    and (session_dir / "thumb").is_dir():
                a_dir = session_dir / "thumb"
            if a_dir is None or not a_dir.is_dir():
                continue
            a_s, a_e = int(src_start[a]), int(src_end[a])
            target_us = np.asarray(ts_mm[a_s:a_e])

            band = _read_band_imu(a_dir / "wavelet_right_wrist" / "wavelet.h5")
            if band is not None:
                ts, imu = band
                idx = _nearest_indices(ts, target_us)
                if not args.dry_run:
                    mmaps["imu_right"][a_s:a_e] = imu[idx]
                n_filled["imu_right"] += len(target_us)

            for fname, sub in (("imu_head", "showee_head"),
                               ("imu_wrist_left", "showee_left_wrist"),
                               ("imu_wrist_right", "showee_right_wrist")):
                jsons = sorted((a_dir / sub).glob("imu_*.json")) \
                    if (a_dir / sub).is_dir() else []
                if len(jsons) != 1:
                    continue
                cam = _read_camera_imu(jsons[0])
                if cam is None:
                    continue
                ts, imu = cam
                idx = _nearest_indices(ts, target_us)
                if not args.dry_run:
                    mmaps[fname][a_s:a_e] = imu[idx]
                n_filled[fname] += len(target_us)
    print("ShowEE fill:", n_filled)

    # ── Incre: weili_imu.csv -> imu_right ──────────────────────────────
    # In the session-level layout the Incre episodes are the last 8 (63..70),
    # in the fixed source order below; their parquet names were dropped by
    # the rebuild.  Only the five sess_* folders have a local weili_imu.csv.
    incre_filled = 0
    incre_order = [
        "data_20260526_172725", "data_20260526_230859", "data_20260527_124150",
        "sess_20260530_102930", "sess_20260530_140912", "sess_20260530_143229",
        "sess_20260531_142701", "sess_20260531_150809",
    ]
    for ep_idx in range(63, min(71, len(target_parquet))):
        name = incre_order[ep_idx - 63]
        sess_dir = _find_dir([args.incre_root], name)
        if sess_dir is None:
            continue
        csvs = sorted((sess_dir / "WeiLiEMG_13_COM3").glob("weili_imu.csv")) \
            if (sess_dir / "WeiLiEMG_13_COM3").is_dir() else []
        if len(csvs) != 1:
            continue
        s = int(target["episode_start_idx"][ep_idx])
        e = int(target["episode_end_idx"][ep_idx])
        if e == s + int(target["episode_length"][ep_idx]) - 1:
            e += 1
        target_us = np.asarray(ts_mm[s:e])
        ts, imu = _read_weili_imu(csvs[0])
        idx = _nearest_indices(ts, target_us)
        if not args.dry_run:
            mmaps["imu_right"][s:e] = imu[idx]
        incre_filled += len(target_us)
        print(f"Incre {name}: {len(target_us):,} rows filled")
    print(f"Incre filled: {incre_filled:,} rows")

    for mm in mmaps.values():
        if mm is not None:
            mm.flush()
    if not args.dry_run:
        shutil.copy2(args.memmap_dir / "manifest.json",
                     args.memmap_dir / "manifest.json.bak3")
        (args.memmap_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2))
    print("done")


if __name__ == "__main__":
    main()
