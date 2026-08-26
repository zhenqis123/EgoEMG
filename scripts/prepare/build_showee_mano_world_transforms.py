#!/usr/bin/env python3
"""Build per-frame MANO-local to mocap-world rigid transforms for ShowEE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from markers2mano.rigid_align import compute_aligned_error_batched
from scripts.mano.infer_mano_for_egoemg import MARKER_VERT_INDICES, load_mano_layer


def open_field(root: Path, manifest: dict, name: str):
    info = manifest["fields"][name]
    return np.memmap(root / info["filename"], mode="r", dtype=info["dtype"], shape=tuple(info["shape"]))


def open_episode_field(root: Path, manifest: dict, name: str):
    info = manifest["episode_fields"][name]
    return np.memmap(root / info["filename"], mode="r", dtype=info["dtype"], shape=tuple(info["shape"]))


def create_field(root: Path, manifest: dict, name: str, total: int):
    filename = f"{name}.dat"
    manifest["fields"][name] = {"filename": filename, "dtype": "float32", "shape": [total, 12]}
    return np.memmap(root / filename, mode="w+", dtype=np.float32, shape=(total, 12))


def nearest_valid_indices(valid: np.ndarray) -> np.ndarray:
    good = np.flatnonzero(valid)
    if len(good) == 0:
        return np.zeros(len(valid), dtype=np.int64)
    pos = np.arange(len(valid))
    ins = np.searchsorted(good, pos)
    left = good[np.maximum(ins - 1, 0)]
    right = good[np.minimum(ins, len(good) - 1)]
    return np.where(pos - left <= right - pos, left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    root = args.memmap_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = np.load(root / "metadata.npz", allow_pickle=False)
    starts, ends = metadata["episode_start_idx"], metadata["episode_end_idx"]
    sources = np.char.decode(metadata["episode_source_parquet"])
    source_root = Path(manifest["source_root"])
    total = int(manifest["total_rows"])
    device = torch.device(args.device)
    mano = load_mano_layer(device)
    marker_indices = MARKER_VERT_INDICES.to(device)
    poses = {h: open_field(root, manifest, f"generated_mano_{h}_pose") for h in ("left", "right")}
    betas = {h: open_episode_field(root, manifest, f"generated_mano_{h}_beta") for h in ("left", "right")}
    outputs = {h: create_field(root, manifest, f"mocap_mano_{h}_world_transform", total) for h in ("left", "right")}

    for episode, (start, end, relative) in enumerate(zip(starts, ends, sources, strict=True)):
        with h5py.File(source_root / relative / "luster_mocap/mocap.h5", "r") as handle:
            markers = {h: handle[f"{h}_hand/markers"][:].astype(np.float32) / 1000.0 for h in ("left", "right")}
        source_count, target_count = len(markers["left"]), int(end - start)
        source_to_target = np.rint(np.linspace(0, target_count - 1, source_count)).astype(np.int64)
        target_to_source = np.rint(np.linspace(0, source_count - 1, target_count)).astype(np.int64)
        for hand in ("left", "right"):
            valid = np.isfinite(markers[hand]).all(axis=(1, 2))
            native = np.zeros((source_count, 12), dtype=np.float32)
            native[:, :9] = np.eye(3, dtype=np.float32).reshape(1, 9)
            valid_indices = np.flatnonzero(valid)
            for offset in range(0, len(valid_indices), args.batch_size):
                indices = valid_indices[offset : offset + args.batch_size]
                pose = torch.from_numpy(np.asarray(poses[hand][start + source_to_target[indices]]).copy()).to(device)
                beta = torch.from_numpy(np.repeat(np.asarray(betas[hand][episode])[None], len(indices), axis=0).copy()).to(device)
                gt = torch.from_numpy(markers[hand][indices]).to(device)
                with torch.no_grad():
                    vertices = mano(pose, beta).verts
                    if hand == "left":
                        # Align in the x-mirrored frame the renderer and the
                        # EgoEMG labels use for left hands (see
                        # ManoMeshDecoder.decode). The previous z-mirror
                        # ([1, 1, -1]) differed by a 180° local-y rotation,
                        # so left meshes rendered upside-down.
                        vertices = vertices * vertices.new_tensor([-1.0, 1.0, 1.0])
                    predicted = vertices[:, marker_indices]
                    _, rotation, translation = compute_aligned_error_batched(predicted, gt)
                native[indices, :9] = rotation.cpu().numpy().reshape(-1, 9)
                native[indices, 9:] = translation.cpu().numpy()
            if valid.any():
                native = native[nearest_valid_indices(valid)]
            outputs[hand][start:end] = native[target_to_source]
        if (episode + 1) % 25 == 0 or episode + 1 == len(starts):
            print(f"[{episode + 1}/{len(starts)}] {relative}", flush=True)
    for output in outputs.values():
        output.flush()
    manifest["mano_world_transform_semantics"] = "row-major R(9)+t(3), local displayed hand mesh to mocap world metres"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done: {root}")


if __name__ == "__main__":
    main()
