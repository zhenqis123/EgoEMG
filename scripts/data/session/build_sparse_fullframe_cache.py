"""Prepare full ego images: all-intra 256px proxy -> decord -> sampled RGB memmap.

No hand crop, mirroring, camera-pose or marker visibility filter is used.
Physical frame ordinals are retained through -vsync 0 and frame_index_map.
python -m scripts.data.session.build_sparse_fullframe_cache --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from egoemg.datasets.session.alignment import (build_alignment, dependency_signature, ego_stream)


def prepare_one(root, entry, interval_us=400_000):
    root = Path(root)
    path = root / entry["path"]
    session = json.loads(path.read_text())
    directory = root / "fullframe_256_index"
    directory.mkdir(exist_ok=True)
    stem = directory / session["session_id"]
    signature = dependency_signature(path, session, interval_us)
    meta = stem.with_suffix(".json")
    if (meta.exists() and stem.with_suffix(".npy").exists()
            and json.loads(meta.read_text())["signature"] == signature):
        return dict(session=session["session_id"], status="cached")
    alignment = build_alignment(path.parent, session, interval_us)
    source = path.parent / ego_stream(session)["video"]["path"]
    proxy = stem.with_suffix(".mp4")
    proxy_signature = [str(source), source.stat().st_size, source.stat().st_mtime_ns]
    receipt = stem.with_suffix(".proxy.json")
    if not (proxy.exists() and receipt.exists()
            and json.loads(receipt.read_text()) == proxy_signature):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to prepare all-intra images")
        temp = stem.with_suffix(".building.mp4")
        command = [
            ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "error",
            "-threads", "2", "-i", str(source), "-map", "0:v:0", "-an",
            "-vf", "scale=256:256:flags=bilinear", "-vsync", "0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-g", "1", "-bf", "0", "-threads", "2", str(temp),
        ]
        subprocess.run(command, check=True)
        # Every proxy frame must be an I frame; validate the encoder result.
        probe = subprocess.check_output([
            shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-skip_frame", "nokey", "-count_frames", "-show_entries",
            "stream=nb_read_frames,nb_frames", "-of", "json", str(temp),
        ])
        stream = json.loads(probe)["streams"][0]
        if int(stream["nb_read_frames"]) != int(stream["nb_frames"]):
            raise RuntimeError(f"Proxy is not all-intra: {temp}")
        temp.replace(proxy)
        receipt.write_text(json.dumps(proxy_signature))
    import decord

    video = decord.VideoReader(str(proxy), ctx=decord.cpu(0), num_threads=1)
    source_frames = subprocess.check_output([
        shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "json", str(source),
    ])
    count = json.loads(source_frames)["streams"][0].get("nb_frames")
    if count not in (None, "N/A") and len(video) != int(count):
        raise RuntimeError(f"Proxy changed physical frame count: {proxy}")
    if len(alignment["physical"]) and alignment["physical"].max() >= len(video):
        raise RuntimeError(f"Physical ordinals exceed proxy length: {proxy}")
    temporary = stem.with_suffix(f".{os.getpid()}.building.npy")
    rgb = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8,
        shape=(len(alignment["frames"]), 256, 256, 3),
    )
    for start in range(0, len(rgb), 32):
        stop = min(start + 32, len(rgb))
        rgb[start:stop] = video.get_batch(alignment["physical"][start:stop]).asnumpy()
    rgb.flush()
    del rgb, video
    with stem.with_suffix(".building.npz").open("wb") as handle:
        np.savez(handle, **alignment)
    temporary.replace(stem.with_suffix(".npy"))
    stem.with_suffix(".building.npz").replace(stem.with_suffix(".npz"))
    temp_meta = meta.with_suffix(".building.json")
    temp_meta.write_text(json.dumps(dict(
        signature=signature, samples=len(alignment["frames"]),
        image_shape=[256, 256, 3], color="RGB", interpolation="bilinear",
        source=str(source), decoder="decord", all_intra_proxy=str(proxy),
        hands=["left", "right"], crop=False, mirror=False,
    )))
    temp_meta.replace(meta)
    return dict(session=session["session_id"], status="complete",
                samples=len(alignment["frames"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/EgoEMG_dataset"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--session")
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    entries = json.loads((root / "dataset.json").read_text())["sessions"]
    if args.session:
        entries = [e for e in entries if e.get("session_id") == args.session
                   or args.session in e["path"]]
        if not entries:
            raise ValueError(args.session)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_one, root, entry) for entry in entries]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    filename = "last_subset_build.json" if args.session else "complete.json"
    (root / "fullframe_256_index" / filename).write_text(
        json.dumps(dict(sessions=len(entries), status="complete"))
    )


if __name__ == "__main__":
    main()
