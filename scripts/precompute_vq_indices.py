#!/usr/bin/env python
"""
Precompute VQ-VAE codebook indices for joint angle sequences and save alongside sessions.
Usage:
  python scripts/precompute_vq_indices.py \
    --config config/vqvae_base.yaml \
    --checkpoint /path/to/vqvae.ckpt \
    --output-dir /path/to/output_dir \
    --hdf5 /path/to/session1.hdf5 /path/to/session2.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from emg2pose.datasets.emg2pose_dataset import Emg2PoseSessionData
from emg2pose.lightning_vqvae import JointAngleVQVAEModule


@torch.no_grad()
def encode_session(
    module: JointAngleVQVAEModule,
    hdf5_path: Path,
    output_path: Path,
    device: torch.device,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, float]:
    with Emg2PoseSessionData(hdf5_path) as session:
        joint_angles_np = session.timeseries[session.JOINT_ANGLES]  # (T, C)
        T, C = joint_angles_np.shape
        if session.no_ik_failure is not None:
            mask_np = session.no_ik_failure[:]
        else:
            mask_np = None

    indices_list: list[torch.Tensor] = []
    mae_acc = 0.0
    count = 0
    pbar = tqdm(total=T, desc=f"Encoding {hdf5_path.name}")
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        angles = torch.as_tensor(joint_angles_np[start:end], dtype=torch.float32, device=device)
        repr_in = module._encode_representation(angles)  # (N, repr_dim)
        z_e = module.model.encoder(repr_in)
        _, idx, vq_loss, codebook_loss, commit_loss = module.model.quantizer(z_e)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)  # (1, N)
        indices_list.append(idx.cpu())

        # reconstruct for error stats, skip NaN targets
        quantized = module.model.quantizer(z_e)[0]
        decoded_repr = module.model.decoder(quantized)
        angles_pred = module._decode_to_angles(decoded_repr)
        valid = torch.isfinite(angles).all(dim=1)
        if valid.any():
            diff = torch.abs(angles_pred[valid] - angles[valid])
            mae_acc += diff.mean().item() * valid.sum().item()
            count += valid.sum().item()
        pbar.update(end - start)
    pbar.close()

    indices = torch.cat(indices_list, dim=1)  # (L, T) or (1, T)
    output = {"indices": indices.long()}
    if mask_np is not None:
        output["mask"] = torch.as_tensor(mask_np, dtype=torch.bool)
    import h5py

    with h5py.File(hdf5_path, "a") as f:
        g = f["emg2pose"]
        if "vq_indices" in g:
            print(f"Skipping write to {hdf5_path} because vq_indices already exists.")
        else:
            g.create_dataset("vq_indices", data=indices.numpy(), compression="gzip")
        if mask_np is not None:
            if "vq_mask" in g:
                print(f"Skipping write of vq_mask to {hdf5_path} because it already exists.")
            else:
                g.create_dataset("vq_mask", data=mask_np, compression="gzip")

    mae = mae_acc / max(count, 1)
    mae_deg = mae * (180.0 / torch.pi)
    print(f"Wrote indices to {hdf5_path} shape={tuple(indices.shape)} | Recon MAE(deg): {mae_deg:.4f}")
    return indices, mae_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute VQ-VAE indices for joint angles.")
    parser.add_argument("--config", type=Path, required=True, help="VQ-VAE config (e.g., config/vqvae_base.yaml)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save indices")
    parser.add_argument("--hdf5-dir", type=Path, required=True, help="Directory containing hdf5 session files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run encoding on")
    parser.add_argument("--chunk-size", type=int, default=65536, help="Chunk size for processing long sequences")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    module_conf = cfg.vqvae
    optimizer_conf = cfg.optimizer
    lr_scheduler_conf = cfg.get("lr_scheduler")
    repr_conf = cfg.angle_representation

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    module = JointAngleVQVAEModule.load_from_checkpoint(
        args.checkpoint,
        vqvae_conf=module_conf,
        optimizer_conf=optimizer_conf,
        lr_scheduler_conf=lr_scheduler_conf,
        repr_conf=repr_conf,
    )
    module.eval().to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_files = sorted(args.hdf5_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise RuntimeError(f"No hdf5 files found in {args.hdf5_dir}")

    for h5 in hdf5_files:
        out_path = args.output_dir / (h5.stem + "_vq_indices.pt")
        encode_session(module, h5, out_path, device=device, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
