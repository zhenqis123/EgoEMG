#!/usr/bin/env python3
"""Build the reusable EgoEMG vision sample index.

This is a one-time preprocessing step for `EgoEmgVisionDataset`. The dataset
loads this sidecar index at startup instead of scanning tens of millions of
frame-level validity entries every time training or visualization starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emg2pose.datasets.egoemg_vision_dataset import build_egoemg_vision_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <memmap-dir>/vision_index.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-episode progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_egoemg_vision_index(
        memmap_dir=args.memmap_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        log_every_episode=not args.quiet,
    )


if __name__ == "__main__":
    main()
