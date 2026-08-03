"""Offline evaluation for fusion models (MidFusion / VisionMLP / ResNet) on EgoEMG.

Reports test metrics per split×hand. The evaluation mask is taken directly from
the model output — the model decides what frames to evaluate (full-window for
``fusion`` mode, center-frame-only for ``center_supervised`` / ``vision_only``).

Usage:
    python -m egoemg.test_analysis_fusion \\
        experiment=fusion/vision_resnet_small_emgfusion_center \\
        --checkpoint logs/fusion/resnet_small_emgfusion_center/version_9/checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoemg.datamodule import make_data_module
from egoemg.lightning import EmgPredictionModule

# Workaround for MKL multi-threading bug on CPU: nn.Linear produces
# incorrect, batch-size-dependent results when batch_size >= 16.
# Single-thread evaluation is correct and deterministic.  GPU is unaffected.
torch.set_num_threads(1)

warnings.filterwarnings(
    "ignore", message="The given NumPy array is not writable", category=UserWarning
)


class _DatasetWithIndex(torch.utils.data.Dataset):
    """Thin wrapper that adds the dataset index to each sample."""

    def __init__(self, base: torch.utils.data.Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        sample["_idx"] = idx
        return sample


def _collate_fusion(batch: list[dict]) -> dict:
    """Collate that includes vision_features / vision_img / _idx for fusion models."""
    from torch.utils.data._utils.collate import default_collate

    emg_batch = []
    for sample in batch:
        emg = sample["emg"]
        if isinstance(emg, np.ndarray):
            emg = torch.as_tensor(emg, dtype=torch.float32)
        ja = sample.get("joint_angles")
        if isinstance(ja, np.ndarray):
            ja = torch.as_tensor(ja, dtype=torch.float32)
        mask = sample.get("label_valid_mask")
        if isinstance(mask, np.ndarray):
            mask = torch.as_tensor(mask, dtype=torch.bool)
        vf = sample.get("vision_features")
        if isinstance(vf, np.ndarray):
            vf = torch.as_tensor(vf, dtype=torch.float32)
        vf_mask = sample.get("vision_valid_mask")
        if isinstance(vf_mask, np.ndarray):
            vf_mask = torch.as_tensor(vf_mask, dtype=torch.bool)
        vi = sample.get("vision_img")
        if isinstance(vi, np.ndarray):
            vi = torch.as_tensor(vi, dtype=torch.float32)

        item = {"emg": emg, "joint_angles": ja, "label_valid_mask": mask}
        if vf is not None:
            item["vision_features"] = vf
        if vf_mask is not None:
            item["vision_valid_mask"] = vf_mask
        if vi is not None:
            item["vision_img"] = vi
        # Pass through _idx if present (for per-group stats)
        if "_idx" in sample:
            item["_idx"] = sample["_idx"]
        emg_batch.append(item)

    return default_collate(emg_batch)


def load_module(ckpt_path: str, config) -> EmgPredictionModule:
    """Load EmgPredictionModule from checkpoint."""
    module = EmgPredictionModule(
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.get("lr_scheduler"),
        loss_weights=config.get("loss_weights", {}),
        datamodule=config.get("datamodule"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k: v for k, v in state_dict.items()}
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    module.eval()
    return module


def _metrics_on_mask(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    metrics_fn: list,
    prefix: str,
) -> dict[str, float]:
    """Compute metrics on a single mask, returning {prefix_mae: val, ...}."""
    if mask is None:
        return {}
    valid = mask.bool()
    if not valid.any():
        return {}
    results: dict[str, float] = {}
    for m in metrics_fn:
        for k, v in m(preds, targets, valid, prefix).items():
            results[k] = float(v)
    return results


def evaluate_dataloader(
    module: EmgPredictionModule,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on one dataloader.

    Uses the mask returned by the model forward pass — the model decides
    which frames to evaluate (single-frame for center_supervised/vision_only,
    full-window for fusion/emg_only).

    Metrics are batch-size-weighted (matching PyTorch Lightning's
    sync_dist=True behaviour) so partial last batches do not bias results.
    """
    module.to(device)
    from egoemg.metrics import get_default_metrics

    metrics_fn = get_default_metrics()
    accum: dict[str, list[tuple[float, int]]] = {}

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            out = module.forward(batch)
            if isinstance(out, tuple):
                preds, targets, mask = out
            else:
                preds = out
                targets = batch["joint_angles"]
                mask = batch["label_valid_mask"]

        n_samples = preds.shape[0]
        m = _metrics_on_mask(preds, targets, mask, metrics_fn, "test")
        for k, v in m.items():
            accum.setdefault(k, []).append((float(v), n_samples))

    avg = {}
    for k, pairs in accum.items():
        values, weights = zip(*pairs)
        avg[k] = float(np.average(values, weights=weights))
    return avg


def evaluate_dataloader_per_group(
    module: EmgPredictionModule,
    dataloader: DataLoader,
    device: torch.device,
    index_meta: list[tuple[str, int]],
) -> tuple[dict, dict]:
    """Evaluate per-user and per-gesture mean ± std in a single pass.

    Collects per-sample predictions, groups by subject and by gesture class,
    computes MAE within each group, then returns per-user and per-gesture stats.
    """
    from egoemg.metrics import AngleMAE

    module.to(device)
    angle_mae_fn = AngleMAE()

    user_groups: dict[str, dict[str, list[torch.Tensor]]] = {}
    gesture_groups: dict[str, dict[str, list[torch.Tensor]]] = {}

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        indices = batch.pop("_idx", None)
        with torch.no_grad():
            out = module.forward(batch)
            if isinstance(out, tuple):
                preds, targets, mask = out
            else:
                preds = out
                targets = batch["joint_angles"]
                mask = batch["label_valid_mask"]

        valid = mask.bool()
        for i in range(preds.shape[0]):
            idx = int(indices[i].item()) if indices is not None else i
            v = valid[i:i + 1]
            if not v.any():
                continue
            subject, gesture = index_meta[idx]
            # per-user
            user_groups.setdefault(subject, {"preds": [], "targets": [], "masks": []})
            user_groups[subject]["preds"].append(preds[i:i + 1].cpu())
            user_groups[subject]["targets"].append(targets[i:i + 1].cpu())
            user_groups[subject]["masks"].append(v.cpu())
            # per-gesture
            gkey = str(gesture)
            gesture_groups.setdefault(gkey, {"preds": [], "targets": [], "masks": []})
            gesture_groups[gkey]["preds"].append(preds[i:i + 1].cpu())
            gesture_groups[gkey]["targets"].append(targets[i:i + 1].cpu())
            gesture_groups[gkey]["masks"].append(v.cpu())

    def _compute_stats(groups: dict, device: torch.device) -> dict:
        per_group_maes = {}
        for key in sorted(groups.keys()):
            g = groups[key]
            if not g["preds"]:
                continue
            p = torch.cat(g["preds"], dim=0).to(device)
            t = torch.cat(g["targets"], dim=0).to(device)
            m = torch.cat(g["masks"], dim=0).to(device)
            mae = float(angle_mae_fn(p, t, m, "test")["test_mae"])
            if not np.isfinite(mae):
                continue
            per_group_maes[key] = mae

        if not per_group_maes:
            return {"count": 0}
        values = list(per_group_maes.values())
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return _compute_stats(user_groups, device), _compute_stats(gesture_groups, device)


def _print_metrics_row(label: str, metrics: dict[str, float]) -> None:
    """Print a compact one-line summary for one metric set."""
    if not metrics:
        print(f"  {label:12s}  N/A (no ground truth available)")
        return
    mae = metrics.get("test_mae", float("nan"))
    ftd = metrics.get("test_landmark/fingertip", float("nan"))
    lmd = metrics.get("test_landmark/all", float("nan"))
    print(f"  {label:12s}  mae={mae:.6f}  fingertip={ftd:.2f}  landmark={lmd:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fusion models on EgoEMG test splits"
    )
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=str, default="fusion_results.csv")
    parser.add_argument(
        "--splits", type=str, nargs="*", default=["user", "gesture", "both"]
    )
    parser.add_argument("--hands", type=str, nargs="*", default=["left", "right"])
    parser.add_argument("--overrides", type=str, nargs="*", default=[])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load config ────────────────────────────────────────────────────────
    config_dir = str(Path(__file__).resolve().parents[1] / "config")

    # Normalise config name: strip optional "experiment/" prefix.
    exp_name = args.config_name
    for prefix in ("experiment/", "experiment\\"):
        if exp_name.startswith(prefix):
            exp_name = exp_name[len(prefix):]
            break

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # Try loading the experiment config directly (works for configs that
        # list `/dataset`, `/transforms` etc without `override`).  If the
        # config uses `override /dataset` (expecting a base config to supply
        # the group), fall back to loading base + experiment override.
        try:
            cfg = compose(config_name=args.config_name, overrides=args.overrides)
        except Exception:
            cfg = compose(
                config_name="base",
                overrides=[f"experiment={exp_name}"] + list(args.overrides),
            )

    has_vision = bool(
        cfg.get("cached_vit_features_dir")
        or cfg.get("per_episode_crops_dir")
        or cfg.module.get("vision_embed_dim")
    )
    print(f"Config: {args.config_name}")
    print(f"Has vision: {has_vision}")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    module = load_module(args.checkpoint, cfg)
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")

    # ── Build datamodule to access dataset config ────────────────────────
    datamodule = make_data_module(cfg)

    memmap_dir = cfg.get("egoemg_memmap_dir")
    emg_layout = cfg.get("egoemg_emg_layout", "emg2pose_interpolate16")
    channel_indices = cfg.get(
        "egoemg_emg2pose_channel_indices", [10, 12, 0, 1, 2, 4, 5, 6]
    )
    channel_interpolate = cfg.get("egoemg_channel_interpolate", False)
    norm_stats_path = cfg.datamodule.get("per_dataset_norm_stats_path")
    val_window = cfg.datamodule.get("val_test_window_length", 7790)
    val_stride = cfg.datamodule.get("val_test_stride", val_window)
    vit_features_dir = cfg.get("cached_vit_features_dir")
    crops_dir = cfg.get("per_episode_crops_dir")
    vision_enabled = bool(
        vit_features_dir or crops_dir or cfg.get("vision_num_frames", 0) > 0
    )
    skip_emg = cfg.get("skip_emg_loading", False)

    # ── Build per-split dataloaders ────────────────────────────────────────
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

    all_results: list[dict] = []

    for split in args.splits:
        for hand in args.hands:
            print(f"\n{'='*50}")
            print(f"Evaluating: split={split}, hand={hand}")

            dataset = EgoEmgMemmapDataset(
                memmap_dir=memmap_dir,
                window_length=int(val_window),
                stride=int(val_stride),
                allowed_splits=[split],
                modalities=["emg", "joint_angles", "labels"],
                target_hand=hand,
                emg_field_preference=cfg.get(
                    "egoemg_emg_field_preference", "filtered"
                ),
                emg_layout=emg_layout,
                emg2pose_channel_indices=channel_indices,
                channel_interpolate=channel_interpolate,
                norm_mode="per-dataset",
                norm_stats_path=norm_stats_path,
                dataset_name="egoemg",
                jitter=False,
                cached_vit_features_dir=vit_features_dir,
                per_episode_crops_dir=crops_dir,
                vision_num_frames=cfg.get("vision_num_frames", 0),
                vision_frame_selection=cfg.get("vision_frame_selection", "center"),
                vision_patch_size=cfg.get("vision_patch_size", 256),
                center_target_only=cfg.get("center_target_only", False),
                skip_emg_loading=skip_emg,
                eval_center_stride=cfg.datamodule.get("eval_center_stride"),
            )

            n = len(dataset)
            print(f"  Samples: {n:,}")
            if n == 0:
                print("  Skipping (empty)")
                continue

            # ── Build index metadata for per-group stats ─────────────
            # Pre-compute subject & gesture for each sample index.
            index_meta: list[tuple[str, int]] = [
                (dataset.get_subject(i), dataset.get_gesture_class(i))
                for i in range(n)
            ]

            # Wrap dataset to include sample index.
            indexed_dataset = _DatasetWithIndex(dataset)
            loader = DataLoader(
                indexed_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,  # must be 0 — index_meta is not pickleable
                pin_memory=(device.type == "cuda"),
                collate_fn=_collate_fusion,
            )

            # ── Overall metrics ──────────────────────────────────────
            metrics = evaluate_dataloader(module, loader, device)

            # ── Per-user & per-gesture mean ± std (single pass) ────
            loader2 = DataLoader(
                indexed_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                collate_fn=_collate_fusion,
            )
            user_stats, gesture_stats = evaluate_dataloader_per_group(
                module, loader2, device, index_meta,
            )

            # ── Print ─────────────────────────────────────────────────
            _print_metrics_row("test", metrics)
            if user_stats.get("count", 0) > 1:
                print(f"  {'per-user':12s}  mae={user_stats['mean']:.6f} ± {user_stats['std']:.6f} "
                      f"(n={user_stats['count']}, min={user_stats['min']:.6f}, max={user_stats['max']:.6f})")
            if gesture_stats.get("count", 0) > 1:
                print(f"  {'per-gesture':12s}  mae={gesture_stats['mean']:.6f} ± {gesture_stats['std']:.6f} "
                      f"(n={gesture_stats['count']}, min={gesture_stats['min']:.6f}, max={gesture_stats['max']:.6f})")

            # ── Collect ────────────────────────────────────────────────
            base = {"split": split, "hand": hand, "samples": n}
            extras = {}
            if user_stats.get("count", 0) > 1:
                extras["per_user_mae_mean"] = user_stats["mean"]
                extras["per_user_mae_std"] = user_stats["std"]
                extras["per_user_count"] = user_stats["count"]
            if gesture_stats.get("count", 0) > 1:
                extras["per_gesture_mae_mean"] = gesture_stats["mean"]
                extras["per_gesture_mae_std"] = gesture_stats["std"]
                extras["per_gesture_count"] = gesture_stats["count"]
            all_results.append(dict(base, **metrics, **extras))

    # ── Save results ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    meta_cols = ["split", "hand", "samples"]
    metric_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + metric_cols]

    print(f"\n{'='*50}")
    print(df.to_string(index=False))
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # ── Summary averages ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Summary (sample-weighted mean over all splits × hands):")
    mae_key = "test_mae"
    if mae_key not in df.columns:
        mae_key = [c for c in df.columns if c.endswith("_mae")][0]
    avg_mae = float(np.average(df[mae_key], weights=df["samples"]))
    print(f"  mean {mae_key} = {avg_mae:.6f}")


if __name__ == "__main__":
    main()
