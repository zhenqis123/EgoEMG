"""Offline 30fps full-ego uint8 RGB memmaps from existing all-intra proxies.

python -m scripts.data.session.build_dense_fullframe_cache --output-dir <ssd-dir>
Original sparse caches and dataset labels are never rewritten.
"""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from egoemg.datasets.session.alignment import (build_alignment, dependency_signature, ego_stream)


def prepare_dense(root, entry, output_dir, fps=30, rgb_path=None):
    import decord

    root, output_dir = Path(root), Path(output_dir)
    path = root / entry["path"]
    session = json.loads(path.read_text())
    stem = output_dir / session["session_id"]
    meta = stem.with_suffix(".json")
    signature = dependency_signature(path, session, 0, fps)
    if (meta.exists() and stem.with_suffix(".npy").exists()
            and json.loads(meta.read_text())["signature"] == signature):
        return dict(session=session["session_id"], status="cached",
                    frames=json.loads(meta.read_text())["samples"])
    alignment = build_alignment(path.parent, session, interval_us=0, target_fps=fps)
    old_stem = root / "fullframe_256_index" / session["session_id"]
    proxy = old_stem.with_suffix(".mp4")
    receipt = old_stem.with_suffix(".proxy.json")
    source = path.parent / ego_stream(session)["video"]["path"]
    expected = [str(source), source.stat().st_size, source.stat().st_mtime_ns]
    if (not proxy.exists() or not receipt.exists()
            or json.loads(receipt.read_text()) != expected):
        raise RuntimeError(f"Prepare verified all-intra proxy first: {proxy}")
    video = decord.VideoReader(str(proxy), ctx=decord.cpu(0), num_threads=1)
    if len(alignment["physical"]) and alignment["physical"].max() >= len(video):
        raise ValueError(f"Invalid physical frame ordinal: {proxy}")
    final_rgb = Path(rgb_path) if rgb_path is not None else stem.with_suffix(".npy")
    final_rgb.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_rgb.with_suffix(".building.npy")
    rgb = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
        shape=(len(alignment["frames"]), 256, 256, 3))
    progress_path = stem.with_suffix(".progress.json")
    try:
        for start in range(0, len(rgb), 128):
            stop = min(start + 128, len(rgb))
            pixels = video.get_batch(alignment["physical"][start:stop]).asnumpy()
            if pixels.shape[1:] != (256, 256, 3):
                raise ValueError("Proxy must be 256x256 RGB")
            rgb[start:stop] = pixels
            if start % 4096 == 0 or stop == len(rgb):
                rgb.flush()
                progress_path.write_text(json.dumps(dict(written=stop, total=len(rgb))))
        rgb.flush()
        # Common frames must be pixel-identical to the existing sparse cache.
        old = np.load(old_stem.with_suffix(".npz"))
        common, old_index, new_index = np.intersect1d(
            old["frames"], alignment["frames"], return_indices=True)
        if len(common):
            choose = np.linspace(0, len(common) - 1, min(16, len(common)), dtype=int)
            previous = np.load(old_stem.with_suffix(".npy"), mmap_mode="r")
            if not np.array_equal(previous[old_index[choose]], rgb[new_index[choose]]):
                raise ValueError("Dense cache differs from sparse RGB at common frames")
        del rgb, video
        with stem.with_suffix(".building.npz").open("wb") as f:
            np.savez(f, **alignment)
        temporary.replace(final_rgb)
        if final_rgb != stem.with_suffix(".npy"):
            link = stem.with_suffix(".building.link")
            link.symlink_to(final_rgb)
            link.replace(stem.with_suffix(".npy"))
        stem.with_suffix(".building.npz").replace(stem.with_suffix(".npz"))
        meta.write_text(json.dumps(dict(signature=signature, samples=len(alignment["frames"]),
            target_fps=fps, image_shape=[256, 256, 3], color="RGB", decoder="decord",
            all_intra_proxy=str(proxy), crop=False, mirror=False,
            common_sparse_pixel_check=True)))
        return dict(session=session["session_id"], status="complete", frames=len(alignment["frames"]))
    except Exception:
        # The incomplete file is deliberately never published as a valid cache.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/EgoEMG_dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rgb-dir", type=Path, default=None,
                        help="Prefer this fast disk; spill to output-dir with reserves")
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads((root / "dataset.json").read_text())["sessions"]
    # This preflight estimate was generated from the actual camera/label index.
    planning = Path("logs/fullframe_dense_cache_planning_2026-10-29.json")
    placement = {}
    if planning.exists() and args.fps == 30:
        plan = json.loads(planning.read_text())
        sizes = {r["session"]: r["fps30_frames"] * 256 * 256 * 3 + 256
                 for r in plan["sessions"]}
        primary = (args.rgb_dir or args.output_dir).resolve()
        primary.mkdir(parents=True, exist_ok=True)
        fast_free = shutil.disk_usage(primary).free - 64 * 2**30
        spill_free = shutil.disk_usage(args.output_dir).free - 32 * 2**30
        for entry in entries:
            sid = json.loads((root / entry["path"]).read_text())["session_id"]
            stem = args.output_dir / sid
            if stem.with_suffix(".json").exists() and stem.with_suffix(".npy").exists():
                placement[sid] = str(stem.with_suffix(".npy").resolve())
                continue
            needed = sizes[sid]
            if fast_free >= needed:
                placement[sid] = str(primary / (sid + ".npy"))
                fast_free -= needed
            elif spill_free >= needed and primary != args.output_dir.resolve():
                placement[sid] = str(args.output_dir.resolve() / (sid + ".npy"))
                spill_free -= needed
            else:
                raise RuntimeError("Insufficient cache storage including disk reserves")
        (args.output_dir / "placement.json").write_text(json.dumps(placement, indent=2))
    results = []
    with (ProcessPoolExecutor(max_workers=max(1, args.workers - 1)) as pool,
          ProcessPoolExecutor(max_workers=1) as spill_pool):
        futures = []
        for entry in entries:
            sid = json.loads((root / entry["path"]).read_text())["session_id"]
            rgb_path = placement.get(sid)
            worker = (spill_pool if rgb_path and Path(rgb_path).parent == args.output_dir.resolve()
                      else pool)
            futures.append(worker.submit(prepare_dense, str(root), entry,
                                         str(args.output_dir), args.fps, rgb_path))
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(completed=len(results), total=len(entries), **result)), flush=True)
    output = dict(status="complete", fps=args.fps, sessions=len(results),
                  frames=sum(r["frames"] for r in results), results=results)
    (args.output_dir / "complete.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
