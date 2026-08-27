#!/usr/bin/env python3
"""Repair + smooth the rigid-body transform streams of the unified memmap.

Two-stage, per-episode pipeline over `mocap_head_transform` and
`mocap_mano_{left,right}_world_transform`:

  1. Transient bridging — removes physically impossible single-step
     teleports (>T_MM mm or >R_RAD rad between consecutive tracked rows)
     and minority-level mis-registration segments (|t| deviating more
     than LEVEL_MM from the episode's median level), by replacing the
     offending rows with SLERP/linear interpolation between the flanking
     stable rows.
  2. Zero-phase Butterworth low-pass (2nd order, 6 Hz at 2 kHz),
     translation directly and rotation via sign-aligned quaternions,
     applied inside contiguous tracked runs only. 6 Hz sits at the
     measured motion/noise spectral knee (gesture content < 4 Hz).

Incre episodes (zero-filled fields) are skipped automatically.

A JSON report with per-episode before/after statistics is written to
--report. Run without --apply for a dry run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot, Slerp

HEAD_T_MM, HEAD_R_RAD = 200.0, 0.30
WORLD_T_MM, WORLD_R_RAD = 50.0, 0.25
LEVEL_MM = 300.0
FC_HZ, FS = 6.0, 2000.0
MERGE_GAP = 500          # rows (~0.25 s)


def steps(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = np.linalg.norm(np.diff(t, axis=0), axis=1) * 1000.0
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
    return dt, ang


def bridge(out_R, out_t, a, b):
    """Interpolate rows (a, b) exclusive between stable endpoints a and b."""
    span = b - a - 1
    if span < 1:
        return
    w = np.linspace(0, 1, span + 2)[1:-1]
    out_t[a + 1:b] = out_t[a] + w[:, None] * (out_t[b] - out_t[a])
    try:
        s = Slerp([0, 1], Rot.from_matrix(np.stack([out_R[a], out_R[b]])))
        out_R[a + 1:b] = s(w).as_matrix()
    except Exception:
        pass


def repair_stream(arr: np.ndarray, ok: np.ndarray, t_mm: float, r_rad: float,
                  is_head: bool) -> tuple[np.ndarray, dict]:
    R = arr[:, :9].reshape(-1, 3, 3).astype(np.float64).copy()
    t = arr[:, 9:12].astype(np.float64).copy()
    n = len(arr)
    before_steps = steps(R, t)
    n_imp_before = int(((before_steps[0] > t_mm) | (before_steps[1] > r_rad)).sum())

    # Nonzero == tracked: the *_valid flags lag/mislabel around
    # mis-registration segments (measured on ep040: valid=False while a
    # nonzero wrong solution is stored), so they must not gate repair.
    ok = np.abs(arr).sum(axis=1) > 0
    if ok.sum() < 100:
        return arr, {"skipped": True, "tracked": int(ok.sum())}

    level = float(np.median(np.linalg.norm(t[ok], axis=1)) * 1000)

    # 1) minority-level mis-registration segments (bridged wholesale)
    wrong = np.abs(np.linalg.norm(t, axis=1) * 1000 - level) > LEVEL_MM
    i = 0
    n_level = 0
    while i < n:
        if wrong[i]:
            j = i
            while j < n and wrong[j]:
                j += 1
            if j - i >= 20 and i >= 1 and j <= n - 1:
                bridge(R, t, i - 1, j)
                n_level += 1
            i = j
        else:
            i += 1

    # 2) impossible seams with return-search: from seam k, extend while the
    # level stays away from the pre-seam median (wrong excursion), up to 2 s.
    HORIZON = 4000
    n_seams = 0
    k = 0
    seams = np.where((steps(R, t)[0] > t_mm) | (steps(R, t)[1] > r_rad))[0]
    for k in seams:
        if k < 1 or k + 1 >= n:
            continue
        pre = np.median(t[max(0, k - 300):k], axis=0)
        m = k + 1
        while m < min(n, k + HORIZON):
            if np.linalg.norm(t[m] - pre) * 1000 < 150:
                break
            m += 1
        if m < min(n, k + HORIZON) and m > k + 1:
            bridge(R, t, k, m)
            n_seams += 1

    # 4) Butterworth 6 Hz zero-phase inside tracked runs
    bcoef, acoef = butter(2, FC_HZ, btype="low", fs=FS)
    q = Rot.from_matrix(R).as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]

    def filt_valid(x):
        out = x.copy()
        i = 0
        while i < n:
            if ok[i]:
                j = i
                while j < n and ok[j]:
                    j += 1
                if j - i > 12:
                    out[i:j] = filtfilt(bcoef, acoef, x[i:j], axis=0)
                i = j
            else:
                i += 1
        return out

    q_s = filt_valid(q)
    q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)
    R_s = Rot.from_quat(q_s).as_matrix()
    t_s = filt_valid(t)

    out = arr.copy()
    out[:, :9] = R_s.reshape(-1, 9)
    out[:, 9:12] = t_s
    out[~ok] = 0.0

    after = steps(R_s, t_s)
    n_imp_after = int(((after[0] > t_mm) | (after[1] > r_rad)).sum())
    return out, {
        "tracked": int(ok.sum()), "level_mm": round(level, 1),
        "level_segments_bridged": n_level, "return_bridges": n_seams,
        "impossible_steps": [n_imp_before, n_imp_after],
        "max_step_mm": [round(float(before_steps[0].max()), 1),
                        round(float(after[0].max()), 1)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--episodes", type=str, default="all",
                    help="'all' or comma-separated indices")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    m = json.loads((args.memmap_dir / "manifest.json").read_text())
    F = m["fields"]

    def load(name, mode):
        s = F[name]
        return np.memmap(args.memmap_dir / s["filename"], dtype=s["dtype"],
                         mode=mode, shape=tuple(s["shape"]))

    ep = load("episode_index", "r")
    lv = load("generated_label_valid", "r").reshape(-1, 2)
    head_valid = load("mocap_head_valid", "r") if "mocap_head_valid" in F else None
    n_eps = int(m["num_episodes"])
    eps = list(range(n_eps)) if args.episodes == "all" else \
        [int(x) for x in args.episodes.split(",")]

    streams = {}
    for name in ("mocap_head_transform", "mocap_mano_left_world_transform",
                 "mocap_mano_right_world_transform"):
        streams[name] = load(name, "r+" if args.apply else "r")

    starts = np.flatnonzero(np.r_[True, np.diff(ep) != 0])
    ends = np.flatnonzero(np.r_[np.diff(ep) != 0, True])
    report = {}
    for e in eps:
        a, b = int(starts[e]), int(ends[e]) + 1
        hv = np.asarray(head_valid[a:b], bool) if head_valid is not None \
            else np.ones(b - a, bool)
        masks = {
            "mocap_head_transform": hv,
            "mocap_mano_left_world_transform": np.asarray(lv[a:b, 0], bool),
            "mocap_mano_right_world_transform": np.asarray(lv[a:b, 1], bool),
        }
        rep = {}
        for name, mask in masks.items():
            arr = np.asarray(streams[name][a:b])
            ok = mask & (np.abs(arr).sum(axis=1) > 0)
            is_head = "head" in name
            out, stats = repair_stream(
                arr, ok,
                HEAD_T_MM if is_head else WORLD_T_MM,
                HEAD_R_RAD if is_head else WORLD_R_RAD,
                is_head)
            if args.apply and not stats.get("skipped"):
                streams[name][a:b] = out.astype(streams[name].dtype)
            rep[name.split("mocap_")[-1][:14]] = stats
        report[f"ep{e:03d}"] = rep
        print(f"ep{e:03d} done", flush=True)

    for s in streams.values():
        if args.apply:
            s.flush()
    args.report.write_text(json.dumps(report, indent=1))
    print(("APPLIED" if args.apply else "DRY RUN") +
          f", report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
