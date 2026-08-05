"""Regenerate the session-level (71-episode) metadata.npz in place.

Fixes two latent inconsistencies left by the session-layout rebuild:

* ``episode_end_idx`` was stored as the LAST row (inclusive,
  ``start + length - 1``) instead of the exclusive bound used by the
  per-action metadata and the dataset window builder.  Convert to
  ``start + length``.
* ``episode_beta_idx`` was all zeros, so every episode read row 0 of the
  per-episode MANO beta memmaps.  Restore the correct rows: EgoEMG
  episodes keep their own row (0..40), each ShowEE session uses the
  beta row of its FIRST action (the session groups that action's
  contiguous rows), and Incre episodes map to their original rows.

Usage::

    python scripts/prepare/rebuild_session_metadata.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_unified_memmap \
        --source-memmap-dir /data/xiziheng/EgoEMG_unified_memmap

Safety: writes ``metadata.npz.bak`` before regenerating.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def _clean(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for v in values:
        s = v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
        s = s.strip("b'").strip('"')
        out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--source-memmap-dir", type=Path, required=True,
                    help="Original per-action (928-episode) memmap whose "
                         "episode order defines the beta row layout.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    path = args.memmap_dir / "metadata.npz"
    md = np.load(path, allow_pickle=False)

    starts = md["episode_start_idx"].astype(np.int64)
    lengths = md["episode_length"].astype(np.int64)
    parquet = _clean(md["episode_source_parquet"])
    beta_idx = md["episode_beta_idx"].astype(np.int32)

    # 1) Exclusive episode_end_idx.
    ends = starts + lengths
    n_changed_end = int((ends != md["episode_end_idx"]).sum())
    print(f"episode_end_idx: {n_changed_end}/{len(starts)} rows changed "
          f"(inclusive -> exclusive)")

    # 2) Beta rows: EgoEMG episodes keep their own row; ShowEE sessions use
    #    the beta row of their first action (from the 928-episode source
    #    metadata); Incre episodes keep their original rows.
    src = np.load(args.source_memmap_dir / "metadata.npz", allow_pickle=False)
    src_parquet = _clean(src["episode_source_parquet"])
    first_action_row: dict[str, int] = {}
    for i, p in enumerate(src_parquet):
        parts = p.split("/")
        if len(parts) == 2 and parts[0] not in first_action_row:
            first_action_row[parts[0]] = i

    new_beta = beta_idx.copy()
    n_changed_beta = 0
    for ep_idx, session in enumerate(parquet):
        if not session:
            continue  # Incre episodes: keep original rows (920..927)
        if session in first_action_row:
            row = int(first_action_row[session])
        else:
            # EgoEMG source episodes: episode index == beta row.
            row = ep_idx
        if new_beta[ep_idx] != row:
            new_beta[ep_idx] = row
            n_changed_beta += 1
    print(f"episode_beta_idx: {n_changed_beta}/{len(new_beta)} rows changed "
          f"(session -> first-action beta row)")

    if args.dry_run:
        print("DRY RUN: no changes written")
        return

    backup = path.with_suffix(".npz.bak")
    if not backup.exists():
        print(f"Backing up -> {backup}")
        shutil.copy2(path, backup)

    out = dict(md)
    out["episode_end_idx"] = ends
    out["episode_beta_idx"] = new_beta
    np.savez(path, **out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
