"""Convert metre-valued 21-marker frames to canonical MANO-right parameters."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .inference import convert_markers, load_model

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m egoemg.markers2mano", description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with markers_m, or an (N,21,3) NPY")
    parser.add_argument("--output", type=Path, required=True, help="New NPZ output path")
    parser.add_argument("--hand", choices=("left", "right"), help="Defaults to NPZ handedness")
    parser.add_argument("--checkpoint", type=Path, default=(
        Path(os.environ.get("EGOEMG_ROOT", ROOT)) / "pretrained_models/markers2mano/markers2mano.ckpt"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    mask, stored_hand, extras = None, None, {}
    if args.input.suffix == ".npz":
        with np.load(args.input, allow_pickle=False) as data:
            if "markers_m" not in data:
                parser.error("NPZ input must contain markers_m with shape (N,21,3)")
            markers = data["markers_m"].copy()
            if "valid_mask" in data:
                mask = data["valid_mask"].copy()
            if "handedness" in data:
                if data["handedness"].ndim != 0:
                    parser.error("NPZ handedness must be a scalar string")
                stored_hand = data["handedness"].item()
                if isinstance(stored_hand, bytes):
                    stored_hand = stored_hand.decode()
            for name in ("timestamps_s", "source_rows"):
                if name in data:
                    extras[name] = data[name].copy()
    elif args.input.suffix == ".npy":
        markers = np.load(args.input, allow_pickle=False)
    else:
        parser.error("--input must be .npz or .npy")
    hand = args.hand or stored_hand
    if hand not in ("left", "right"):
        parser.error("Specify --hand or include scalar handedness in the input NPZ")
    if markers.ndim != 3 or markers.shape[1:] != (21, 3):
        parser.error("Expected markers_m with shape (N,21,3), in metres")
    if mask is not None and mask.shape not in ((len(markers),), (len(markers), 21)):
        parser.error("valid_mask must have shape (N,21) or (N,)")
    if args.hand is not None and stored_hand is not None and args.hand != stored_hand:
        parser.error("--hand differs from the input NPZ handedness")
    for name, values in extras.items():
        if values.shape != (len(markers),):
            parser.error(f"{name} must have shape (N,)")
    torch.set_num_threads(4)
    model = load_model(args.checkpoint, args.device)
    result = convert_markers(model, markers, handedness=hand,
                             valid_mask=mask, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez(stream, **result, **extras, handedness=hand, model_name="markers2mano")
    print(f"Converted {int(result['valid'].sum())}/{len(markers)} frames: {args.output}")


if __name__ == "__main__":
    main()
