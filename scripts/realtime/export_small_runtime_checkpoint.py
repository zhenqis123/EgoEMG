#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from emg2pose.realtime_local.small_model import _extract_state_dict, _strip_model_prefix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Lightning small EMGFormer checkpoint to a runtime-only state dict."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        ckpt = torch.load(
            Path(args.checkpoint).expanduser(),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    state = _strip_model_prefix(_extract_state_dict(ckpt))
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "format": "emg2pose_small_runtime_v1"}, output)
    print(f"Wrote {output} ({len(state)} tensors)")


if __name__ == "__main__":
    main()
