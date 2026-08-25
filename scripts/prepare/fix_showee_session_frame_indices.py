"""Rewrite ShowEE ``image_head_frame_index`` for session-level videos.

Background
----------
The unified memmap stores ShowEE rows with PER-ACTION local webcam frame
indices: each action's raw ``showee_head/*.mkv`` restarts at frame 0, and
``image_head_frame_index`` was built as the index into that local video.

The released all-intra videos are SESSION-level concatenations
(``episode_000041..062_head_allintra.mp4`` — each session's ~40 action
recordings joined end-to-end, verified to have exactly
``sum(mkv_frames)`` frames).  Reading local indices against the session
video therefore drifts by the cumulative frame count of all preceding
actions — the "video offset" bug.

This script rewrites ``image_head_frame_index.dat`` in place so every
ShowEE row indexes into its session video::

    new[i] = local[i] + sum(mkv_frames[action_0 .. action_{k-1}])

for the action ``k`` containing row ``i``.  Action boundaries come from
the ORIGINAL per-action (928-episode) metadata, and local values are read
from the untouched original memmap — no heuristic frame-index wrap
detection, so the result is deterministic and idempotent.  This replaces
the ad-hoc ``/tmp/fix_offsets.py`` which (a) inferred boundaries from
frame-index wraps and (b) assigned the first row of each action (local
index 0) to the previous segment, leaving ~1 row per action pointing at
the previous action's first frame.

Usage::

    python scripts/prepare/fix_showee_session_frame_indices.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_full_memmap \
        --source-memmap-dir /data/xiziheng/EgoEMG_full_memmap \
        --showee-root /mnt/nvme/xiziheng \
        --allintra-root /mnt/nvme/xiziheng/EgoEMG_videos

Safety: backs up ``image_head_frame_index.dat`` to ``.bak`` before
writing (unless ``--dry-run``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _clean(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for v in values:
        s = v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
        s = s.strip("b'").strip('"')
        out.append(s)
    return out


def _load_metadata(memmap_dir: Path) -> dict:
    md = np.load(memmap_dir / "metadata.npz", allow_pickle=False)
    return {
        "episode_id": _clean(md["episode_id"]),
        "source_parquet": _clean(md["episode_source_parquet"]),
        "head_video_path": _clean(md["episode_head_video_path"]),
        "start": md["episode_start_idx"].astype(np.int64),
        "end": md["episode_end_idx"].astype(np.int64),
        "length": md["episode_length"].astype(np.int64),
    }


def _mkv_frame_count(mkv_path: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(mkv_path)],
        capture_output=True, text=True, check=True,
    )
    return int(r.stdout.strip())


def _video_frame_count(video_path: Path) -> int:
    return _mkv_frame_count(video_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memmap-dir", type=Path, required=True,
                    help="Session-level (71-episode) memmap whose frame "
                         "indices are rewritten.")
    ap.add_argument("--source-memmap-dir", type=Path, required=True,
                    help="Original per-action (928-episode) memmap: exact "
                         "action boundaries and untouched local frame indices.")
    ap.add_argument("--showee-root", type=Path, nargs="+", required=True,
                    help="Directories containing the raw ShowEE session folders "
                         "(e.g. /mnt/nvme/xiziheng/showee/downloads "
                         "/mnt/nvme/xiziheng). Session dirs are searched in order.")
    ap.add_argument("--allintra-root", type=Path, required=True,
                    help="Flat all-intra video dir (episode_000041_allintra.mp4 ...).")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _find_session_dir(roots: list[Path], session_name: str) -> Path | None:
    for root in roots:
        p = root / session_name
        if p.is_dir():
            return p
    return None


# Metadata action names that differ from the on-disk directory names
# (the raw session folders were renamed when packaged into .7z).
ACTION_DIR_ALIASES = {
    "thum": "thumb",
}


def _find_action_dir(session_dir: Path, action_name: str) -> Path | None:
    direct = session_dir / action_name
    if direct.is_dir():
        return direct
    alias = ACTION_DIR_ALIASES.get(action_name)
    if alias is not None:
        aliased = session_dir / alias
        if aliased.is_dir():
            print(f"    (action '{action_name}' found on disk as '{alias}')")
            return aliased
    return None


def main() -> None:
    args = parse_args()
    showee_roots = args.showee_root

    target = _load_metadata(args.memmap_dir)
    source = _load_metadata(args.source_memmap_dir)

    # ShowEE sessions in the target metadata = episodes whose source parquet
    # is a directory under one of the showee roots (the 71-episode layout
    # groups each session into one episode).
    session_eps = []
    for i, parquet in enumerate(target["source_parquet"]):
        if parquet and _find_session_dir(showee_roots, parquet) is not None:
            session_eps.append(i)
    print(f"Found {len(session_eps)} ShowEE session episodes in target metadata")

    # ShowEE actions in the source metadata = episodes whose parquet is
    # "<session>/<action>".
    action_by_session: dict[str, list[int]] = {}
    for i, parquet in enumerate(source["source_parquet"]):
        parts = parquet.split("/")
        if len(parts) == 2 and _find_session_dir(showee_roots, parts[0]) is not None:
            action_by_session.setdefault(parts[0], []).append(i)
    print(f"Found {sum(len(v) for v in action_by_session.values())} actions "
          f"across {len(action_by_session)} sessions in source metadata")

    with (args.memmap_dir / "manifest.json").open() as f:
        manifest = json.load(f)
    fi_info = manifest["fields"]["image_head_frame_index"]
    total_rows = int(fi_info["shape"][0])

    dat_path = args.memmap_dir / fi_info["filename"]
    if args.dry_run:
        fi = np.memmap(dat_path, dtype=fi_info["dtype"], mode="r",
                       shape=(total_rows,))
    else:
        backup = dat_path.with_suffix(".dat.bak")
        if not backup.exists():
            print(f"Backing up -> {backup}")
            shutil.copy2(dat_path, backup)
        fi = np.memmap(dat_path, dtype=fi_info["dtype"], mode="r+",
                       shape=(total_rows,))
    src_fi = np.memmap(
        args.source_memmap_dir / fi_info["filename"],
        dtype=fi_info["dtype"], mode="r", shape=(total_rows,),
    )

    report = {}
    for ep_idx in tqdm(session_eps, desc="sessions", unit="ep"):
        session_name = target["source_parquet"][ep_idx]
        # The 71-episode metadata stores episode_end_idx as the LAST row
        # (inclusive); convert to the exclusive bound used by the source
        # per-action metadata.
        s = int(target["start"][ep_idx])
        e = int(target["end"][ep_idx])
        # Legacy metadata stored the LAST row (inclusive); regenerated
        # metadata uses the exclusive bound.
        if e == s + int(target["length"][ep_idx]) - 1:
            e += 1
        session_dir = _find_session_dir(showee_roots, session_name)
        assert session_dir is not None

        actions = sorted(
            action_by_session.get(session_name, []),
            key=lambda i: int(source["start"][i]),
        )
        if not actions:
            report[session_name] = {"error": "no actions in source metadata"}
            print(f"  {session_name}: no actions found, skipping")
            continue

        # Verify the actions tile the session range exactly.
        ranges = [(int(source["start"][a]), int(source["end"][a])) for a in actions]
        if ranges[0][0] != s or ranges[-1][1] != e:
            report[session_name] = {
                "error": "range mismatch",
                "session": [s, e],
                "actions": ranges[:3],
                "actions_last": ranges[-1],
            }
            print(f"  {session_name}: session range [{s},{e}) not covered by actions "
                  f"[{ranges[0][0]},{ranges[-1][1]}), skipping")
            continue
        if any(ranges[i][1] != ranges[i + 1][0] for i in range(len(ranges) - 1)):
            report[session_name] = {"error": "non-contiguous actions"}
            print(f"  {session_name}: actions are not contiguous, skipping")
            continue

        # Per-action mkv frame counts (session video = exact concatenation of
        # the action recordings that HAVE a webcam stream).
        frame_counts = []
        bad = False
        for a in actions:
            action_name = source["source_parquet"][a].split("/")[1]
            a_s, a_e = int(source["start"][a]), int(source["end"][a])
            action_dir = _find_action_dir(session_dir, action_name)
            mkv_files = (
                sorted((action_dir / "showee_head").glob("*.mkv"))
                if action_dir is not None else []
            )
            if len(mkv_files) == 0:
                # No webcam recording for this action.  The memmap rows must
                # be fully stale (-1) — otherwise the session video (which
                # omits this action) could not cover them.
                local = np.asarray(src_fi[a_s:a_e])
                if local.max() >= 0:
                    report[session_name] = {
                        "error": f"no mkv but valid frame indices for {action_name}",
                    }
                    print(f"  {session_name}/{action_name}: no mkv but memmap has "
                          f"valid frame indices, skipping")
                    bad = True
                    break
                frame_counts.append(0)
                continue
            if len(mkv_files) != 1:
                report[session_name] = {
                    "error": f"mkv count {len(mkv_files)} for {action_name}",
                }
                print(f"  {session_name}/{action_name}: expected 1 mkv, got "
                      f"{len(mkv_files)}, skipping")
                bad = True
                break
            frame_counts.append(_mkv_frame_count(mkv_files[0]))
        if bad:
            continue

        offsets = np.concatenate([[0], np.cumsum(frame_counts)[:-1]])
        video_path = args.allintra_root / (
            target["head_video_path"][ep_idx].replace(".mp4", "_allintra.mp4")
        )
        if not video_path.is_file():
            report[session_name] = {"error": f"missing session video {video_path}"}
            print(f"  {session_name}: missing session video {video_path}, skipping")
            continue
        video_frames = _video_frame_count(video_path)

        n_rewritten = 0
        for a, off in zip(actions, offsets):
            a_s, a_e = int(source["start"][a]), int(source["end"][a])
            local = np.asarray(src_fi[a_s:a_e])
            mask = local >= 0
            n_rewritten += int(mask.sum())
            if not args.dry_run:
                fi[a_s:a_e][mask] = local[mask] + off

        if not args.dry_run:
            fi.flush()
        new_vals = np.asarray(fi[s:e])
        nv = new_vals[new_vals >= 0]
        wraps = int((np.diff(nv) < 0).sum()) if len(nv) else 0
        total_frames = int(sum(frame_counts))
        report[session_name] = {
            "ok": total_frames == video_frames,
            "actions": len(actions),
            "rows": int(e - s),
            "rows_rewritten": n_rewritten,
            "session_video_frames": video_frames,
            "sum_mkv_frames": total_frames,
            "max_frame_index": int(nv.max()) if len(nv) else -1,
            "wraps_remaining": wraps,
            "first_offsets": [int(o) for o in offsets[:4]],
        }
        status = "✓" if report[session_name]["ok"] and wraps == 0 else "⚠"
        print(f"  {session_name}: {len(actions)} actions, video={video_frames} "
              f"sum_mkv={total_frames} max_fi={nv.max() if len(nv) else -1} "
              f"wraps={wraps} {status}")

    out = args.memmap_dir / "frame_index_fix_report.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
