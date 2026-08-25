#!/usr/bin/env python3
"""Batch re-encode EgoEMG webcam videos into all-intra H.264 files.

This script reads webcam video paths from the EgoEMG memmap metadata and
produces a mirrored directory tree of re-encoded videos optimized for random
frame access. The output uses H.264 with every frame forced to be an I-frame,
which is a practical trade-off between seek performance and storage growth for
OpenCV-based visualization workflows.

Example:
    python scripts/prepare/reencode_egoemg_webcam_allintra.py \
        --memmap-dir data/EgoEMG_memmap \
        --data-root data/EgoEMG \
        --output-root data/EgoEMG_videos \
        --jobs 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DEFAULT_MEMMAP_DIR = Path("data/EgoEMG_memmap")
DEFAULT_DATA_ROOT = Path("data/EgoEMG")
DEFAULT_OUTPUT_ROOT = Path("data/EgoEMG_videos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        default=DEFAULT_MEMMAP_DIR,
        help="Path to EgoEMG memmap directory containing metadata.npz.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory used to resolve relative webcam video paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for re-encoded videos. Relative layout is preserved.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_allintra",
        help="Suffix added before the output file extension.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of ffmpeg jobs to run in parallel.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="ultrafast",
        help="x264 preset for all-intra encoding.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF for all-intra encoding. Lower is larger / higher quality.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N unique videos. Useful for testing.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable name or absolute path.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Optional JSON file recording source-to-output path mapping.",
    )
    return parser.parse_args()


def decode_bytes(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            decoded.append(value.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(value))
    return decoded


def load_video_paths(memmap_dir: Path) -> list[str]:
    metadata_path = memmap_dir / "metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = np.load(metadata_path, allow_pickle=False)
    if "episode_webcam_video_path" not in metadata:
        raise KeyError("metadata.npz does not contain 'episode_webcam_video_path'")

    return decode_bytes(metadata["episode_webcam_video_path"])


def resolve_source_path(data_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return data_root / path


def build_output_path(
    source_path: Path,
    data_root: Path,
    output_root: Path,
    suffix: str,
) -> Path:
    try:
        rel_path = source_path.relative_to(data_root)
    except ValueError:
        rel_path = Path(source_path.name)
    return output_root / rel_path.with_name(f"{rel_path.stem}{suffix}.mp4")


def ensure_ffmpeg(ffmpeg_bin: str) -> None:
    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg_bin}")


def encode_one(
    ffmpeg_bin: str,
    source_path: Path,
    output_path: Path,
    preset: str,
    crf: int,
    overwrite: bool,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "source": str(source_path),
            "output": str(output_path),
            "size_bytes": output_path.stat().st_size,
        }

    cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-tune",
        "fastdecode",
        "-x264-params",
        "keyint=1:min-keyint=1:scenecut=0:bframes=0:ref=1:cabac=0",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_s = time.perf_counter() - start

    if proc.returncode != 0:
        return {
            "status": "failed",
            "source": str(source_path),
            "output": str(output_path),
            "elapsed_s": elapsed_s,
            "stderr": proc.stderr.strip(),
        }

    return {
        "status": "encoded",
        "source": str(source_path),
        "output": str(output_path),
        "elapsed_s": elapsed_s,
        "size_bytes": output_path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    ensure_ffmpeg(args.ffmpeg_bin)

    raw_paths = load_video_paths(args.memmap_dir)
    unique_raw_paths = list(dict.fromkeys(raw_paths))
    if args.limit is not None:
        unique_raw_paths = unique_raw_paths[: args.limit]

    jobs = max(1, args.jobs)
    print(f"Found {len(unique_raw_paths)} unique webcam videos")
    print(f"Input root:  {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Jobs:        {jobs}")
    print(f"Preset/CRF:  {args.preset} / {args.crf}")

    tasks: list[tuple[Path, Path]] = []
    missing_sources: list[str] = []
    for raw_path in unique_raw_paths:
        if not raw_path.strip():
            missing_sources.append("<empty webcam path>")
            continue
        source_path = resolve_source_path(args.data_root, raw_path)
        if not source_path.is_file():
            missing_sources.append(str(source_path))
            continue
        output_path = build_output_path(
            source_path=source_path,
            data_root=args.data_root,
            output_root=args.output_root,
            suffix=args.suffix,
        )
        tasks.append((source_path, output_path))

    if missing_sources:
        print(f"Warning: {len(missing_sources)} source videos are missing and will be skipped")
        for missing in missing_sources[:10]:
            print(f"  missing: {missing}")
        if len(missing_sources) > 10:
            print(f"  ... and {len(missing_sources) - 10} more")

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                encode_one,
                args.ffmpeg_bin,
                source_path,
                output_path,
                args.preset,
                args.crf,
                args.overwrite,
            )
            for source_path, output_path in tasks
        ]

        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            status = str(result["status"])
            source = str(result["source"])
            output = str(result["output"])
            if status == "failed":
                print(f"[{idx}/{len(tasks)}] failed   {source}")
                stderr = str(result.get("stderr", ""))
                if stderr:
                    print(stderr)
            elif status == "skipped":
                print(f"[{idx}/{len(tasks)}] skipped  {output}")
            else:
                elapsed_s = float(result["elapsed_s"])
                print(f"[{idx}/{len(tasks)}] encoded  {output}  ({elapsed_s:.1f}s)")

    encoded = [r for r in results if r["status"] == "encoded"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    total_size = sum(int(r.get("size_bytes", 0)) for r in encoded + skipped)

    print("")
    print("Summary")
    print(f"  encoded: {len(encoded)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed:  {len(failed)}")
    print(f"  output bytes (existing + new): {total_size}")

    if args.write_manifest is not None:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "data_root": str(args.data_root),
            "output_root": str(args.output_root),
            "suffix": args.suffix,
            "preset": args.preset,
            "crf": args.crf,
            "results": results,
        }
        with open(args.write_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote manifest: {args.write_manifest}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
