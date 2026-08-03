"""Measure EMG delta contribution ratio for a center_supervised fusion checkpoint.

Usage:
  python scripts/eval/check_delta_contribution.py \
    experiment=fusion/vision_resnet_small_emgfusion_center \
    checkpoint=/path/to/checkpoint.ckpt
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from egoemg.datamodule import make_data_module
from egoemg.lightning import EmgPredictionModule

log = logging.getLogger(__name__)


def analyze_delta(model, datamodule, device: str = "cpu") -> dict:
    """Run validation data through model and compute delta contribution stats."""
    model.eval()
    model.to(device)

    val_loaders = datamodule.val_dataloader()
    if not isinstance(val_loaders, list):
        val_loaders = [val_loaders]

    all_ratios = []       # |delta| / |pred| per sample
    all_delta_norms = []  # mean(|delta|) per sample
    all_pred_norms = []   # mean(|pred|) per sample
    all_yv_norms = []     # mean(|y_v|) per sample
    n_samples = 0

    for loader in val_loaders:
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.no_grad():
                output = model.model(batch)

            delta = getattr(model.model, "_last_delta", None)
            if delta is None:
                log.warning("No _last_delta found on model — not a fusion model?")
                continue

            # Hook to capture y_v: re-run head_vision
            if "vision_features" in batch:
                vf = batch["vision_features"]
            elif "vision_img" in batch:
                vf = model.model._extract_vision_features(batch["vision_img"])
            else:
                continue
            y_v = model.model.head_vision(vf)  # (B, 22)

            if isinstance(output, tuple):
                preds = output[0]
            else:
                preds = output

            delta_flat = delta.squeeze(-1)  # (B, 22)
            preds_flat = preds.squeeze(-1)  # (B, 22)

            # Per-sample mean absolute value across 22 joints
            delta_mean = delta_flat.abs().mean(dim=-1)  # (B,)
            pred_mean = preds_flat.abs().mean(dim=-1)
            yv_mean = y_v.abs().mean(dim=-1)

            ratio = delta_mean / (pred_mean + 1e-8)

            all_ratios.extend(ratio.cpu().tolist())
            all_delta_norms.extend(delta_mean.cpu().tolist())
            all_pred_norms.extend(pred_mean.cpu().tolist())
            all_yv_norms.extend(yv_mean.cpu().tolist())
            n_samples += preds.shape[0]

            if n_samples % 2000 == 0:
                log.info("Processed %d samples...", n_samples)

    ratio_t = torch.tensor(all_ratios)
    delta_t = torch.tensor(all_delta_norms)
    pred_t = torch.tensor(all_pred_norms)
    yv_t = torch.tensor(all_yv_norms)

    return {
        "n_samples": n_samples,
        # |delta| / |pred|
        "ratio_mean": float(ratio_t.mean()),
        "ratio_median": float(ratio_t.median()),
        "ratio_std": float(ratio_t.std()),
        "ratio_p10": float(ratio_t.quantile(0.10)),
        "ratio_p90": float(ratio_t.quantile(0.90)),
        # |delta| / |y_v|
        "ratio_vs_yv_mean": float((delta_t / (yv_t + 1e-8)).mean()),
        "ratio_vs_yv_median": float((delta_t / (yv_t + 1e-8)).median()),
        # Magnitudes
        "delta_mean": float(delta_t.mean()),
        "pred_mean": float(pred_t.mean()),
        "yv_mean": float(yv_t.mean()),
        "delta_rms": float((delta_t ** 2).mean() ** 0.5),
        "pred_rms": float((pred_t ** 2).mean() ** 0.5),
    }


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def main(config: DictConfig):
    log.info("Building model and datamodule from config...")

    checkpoint_path = config.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Must specify --config checkpoint=/path/to/checkpoint.ckpt")

    # Build module
    module = EmgPredictionModule(
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        loss_weights=config.loss_weights,
        component_lr_scales=config.get("component_lr_scales"),
        datamodule=config.datamodule,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    log.info("Loaded checkpoint (missing=%d, unexpected=%d)", len(missing), len(unexpected))

    epoch = ckpt.get("epoch", "?")
    log.info("Checkpoint epoch: %s", epoch)

    # Build datamodule
    datamodule = make_data_module(config)

    stats = analyze_delta(module, datamodule)

    print(f"\n{'='*60}")
    print(f"Delta Contribution Analysis")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {epoch}")
    print(f"Samples: {stats['n_samples']}")
    print(f"{'='*60}")

    print(f"\n--- Per-sample |delta| / |pred| ratio ---")
    print(f"  Mean:   {stats['ratio_mean']:.4f}  ({stats['ratio_mean']*100:.1f}%)")
    print(f"  Median: {stats['ratio_median']:.4f}  ({stats['ratio_median']*100:.1f}%)")
    print(f"  Std:    {stats['ratio_std']:.4f}")
    print(f"  P10:    {stats['ratio_p10']:.4f}")
    print(f"  P90:    {stats['ratio_p90']:.4f}")

    print(f"\n--- Per-sample |delta| / |y_v| ratio ---")
    print(f"  Mean:   {stats['ratio_vs_yv_mean']:.4f}  ({stats['ratio_vs_yv_mean']*100:.1f}%)")
    print(f"  Median: {stats['ratio_vs_yv_median']:.4f}  ({stats['ratio_vs_yv_median']*100:.1f}%)")

    print(f"\n--- Absolute magnitudes (mean |x| across joints, then mean across samples) ---")
    print(f"  |y_v|   mean: {stats['yv_mean']:.4f}")
    print(f"  |delta| mean: {stats['delta_mean']:.4f}")
    print(f"  |pred|  mean: {stats['pred_mean']:.4f}")
    print(f"  |delta| / |y_v|  mean-of-means: {stats['delta_mean'] / stats['yv_mean']:.4f}")
    print(f"  |delta| / |pred| mean-of-means: {stats['delta_mean'] / stats['pred_mean']:.4f}")


if __name__ == "__main__":
    main()
