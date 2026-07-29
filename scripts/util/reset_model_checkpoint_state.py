#!/usr/bin/env python3
"""Copy a Lightning checkpoint while dropping ModelCheckpoint callback state."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    callbacks = checkpoint.get("callbacks", {})
    removed = [key for key in callbacks if "ModelCheckpoint" in str(key)]
    for key in removed:
        del callbacks[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Removed {len(removed)} ModelCheckpoint state entries: {removed}")
    print(f"Saved sanitized checkpoint: {args.output}")


if __name__ == "__main__":
    main()
