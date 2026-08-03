"""Per-gesture analysis of fusion checkpoints.

Evaluates a fusion checkpoint on test data, computes vision-only, EMG-only,
and fusion MAE for each sample, groups results by gesture class, and reports
per-gesture statistics.

Usage:
    python scripts/eval/analyze_per_gesture.py \
        --checkpoint logs/fusion/resnet_small_emgfusion_center/version_14/checkpoints/resnet-small-centerfusion-epoch=137-val_mae=0.0945.ckpt \
        --output-dir ./per_gesture_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from egoemg.lightning import EmgPredictionModule
from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from egoemg.models.modules.emgformer import Emg2PoseFormer
from egoemg.models.modules.vit_vision import VisionViTPose
from egoemg.models.modules.resnet_vision import ResNetVisionPose

CONFIG_DIR = str(_PROJECT_DIR / "config")

warnings.filterwarnings("ignore", message="enable_nested_tensor is True", category=UserWarning)


def load_module(ckpt_path: str, config) -> EmgPredictionModule:
    module = EmgPredictionModule(
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.get("lr_scheduler"),
        loss_weights=config.get("loss_weights", {}),
        datamodule=config.get("datamodule"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    module.eval()
    return module


def evaluate_per_gesture(
    module: EmgPredictionModule,
    dataloader: DataLoader,
    dataset: EgoEmgMemmapDataset,
    device: torch.device,
    emg_model: Emg2PoseFormer | None = None,
    vis_model: VisionViTPose | None = None,
) -> list[dict]:
    """Evaluate all samples, returning per-sample dicts with gesture class."""
    module.to(device)
    results: list[dict] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        abs_indices = batch.get("_abs_idx")

        with torch.no_grad():
            # Fusion forward
            out = module.forward(batch_gpu)
            if isinstance(out, tuple):
                preds, targets, mask = out
            else:
                preds = out
                targets = batch_gpu["joint_angles"]
                mask = batch_gpu["label_valid_mask"]

            preds_fusion = preds.squeeze(-1)
            if targets.ndim == 3:
                targets = targets.squeeze(-1)

            # Vision-only forward
            if vis_model is not None and "vision_img" in batch_gpu:
                vis_out = vis_model(batch_gpu)
                y_v_all = vis_out[0].squeeze(-1) if isinstance(vis_out, tuple) else vis_out.squeeze(-1)
            else:
                y_v_all = torch.zeros(preds_fusion.shape[0], preds_fusion.shape[1], device=device)

            # EMG-only forward
            if emg_model is not None:
                emg_out = emg_model(batch_gpu)
                emg_preds_full = emg_out[0] if isinstance(emg_out, tuple) else emg_out
                t_center = emg_preds_full.shape[-1] // 2
                y_emg_all = emg_preds_full[:, :, t_center]
            else:
                y_emg_all = torch.zeros(preds_fusion.shape[0], preds_fusion.shape[1], device=device)

        for i in range(preds_fusion.shape[0]):
            mae_vision = float((y_v_all[i] - targets[i]).abs().mean())
            mae_fusion = float((preds_fusion[i] - targets[i]).abs().mean())
            mae_emg = float((y_emg_all[i] - targets[i]).abs().mean())

            abs_idx = int(abs_indices[i]) if abs_indices is not None else -1
            gesture_class = dataset.get_gesture_class(abs_idx) if abs_idx >= 0 else -1

            results.append({
                "mae_vision": mae_vision,
                "mae_fusion": mae_fusion,
                "mae_emg": mae_emg,
                "gesture_class": gesture_class,
                "abs_idx": abs_idx,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Per-gesture fusion checkpoint analysis")
    parser.add_argument("--config-name", required=True,
                        help="e.g. experiment/fusion/vision_resnet_small_emgfusion_center")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-location", default="./data/EgoEMG_memmap")
    parser.add_argument("--splits", nargs="+", default=["gesture"])
    parser.add_argument("--hands", nargs="+", default=["left", "right"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="./per_gesture_analysis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per split/hand (0=all)")
    parser.add_argument("--emg-checkpoint", default=None,
                        help="Standalone EMG checkpoint path (overrides config)")
    parser.add_argument("--vision-checkpoint", default=None,
                        help="Standalone vision checkpoint path (overrides config)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────
    exp_name = args.config_name
    for prefix in ("experiment/", "experiment\\"):
        if exp_name.startswith(prefix):
            exp_name = exp_name[len(prefix):]
            break

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        try:
            cfg = compose(config_name=args.config_name)
        except Exception:
            cfg = compose(config_name="base", overrides=[f"experiment={exp_name}"])

    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    # ── Load fusion model ──────────────────────────────────────────────
    print("\nLoading fusion checkpoint...")
    module = load_module(args.checkpoint, cfg)
    module.to(device)
    module.eval()

    # ── Load standalone EMG model ──────────────────────────────────────
    emg_ckpt_path = args.emg_checkpoint or cfg.get("pretrained_emg_checkpoint")
    emg_model = None
    if emg_ckpt_path and Path(emg_ckpt_path).exists():
        from hydra.utils import instantiate
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
            try:
                emg_cfg = compose(config_name="experiment/emgformer/regression_emgformer_small_aggressive_egoemg")
            except Exception:
                emg_cfg = compose(config_name="base", overrides=["experiment=emgformer/regression_emgformer_small_aggressive_egoemg"])
        emg_model = Emg2PoseFormer(
            featurizer=instantiate(emg_cfg.module.featurizer),
            decoder=instantiate(emg_cfg.module.decoder),
            head=instantiate(emg_cfg.module.head),
            out_channels=emg_cfg.module.get("out_channels", 22),
            provide_initial_pos=emg_cfg.module.get("provide_initial_pos", False),
            center_supervised=False,
        )
        emg_ckpt = torch.load(emg_ckpt_path, map_location="cpu", weights_only=False)
        emg_sd = emg_ckpt.get("state_dict", emg_ckpt)
        emg_sd_remapped = {k[6:] if k.startswith("model.") else k: v for k, v in emg_sd.items()}
        emg_model.load_state_dict(emg_sd_remapped, strict=False)
        emg_model.to(device)
        emg_model.eval()
        print(f"Loaded standalone EMG model: {sum(p.numel() for p in emg_model.parameters()):,} params")

    # ── Load standalone vision model ───────────────────────────────────
    vis_ckpt_key = "vision_vit_checkpoint" if cfg.module.get("vision_backbone_type", "").startswith("vit") else "vision_resnet_checkpoint"
    vis_ckpt_path = args.vision_checkpoint or cfg.get(vis_ckpt_key)
    vis_model = None
    if vis_ckpt_path and Path(vis_ckpt_path).exists():
        vis_ckpt = torch.load(vis_ckpt_path, map_location="cpu", weights_only=False)
        vis_sd = vis_ckpt.get("state_dict", vis_ckpt)
        vis_sd_remapped = {k[6:] if k.startswith("model.") else k: v for k, v in vis_sd.items()}
        vis_backbone = cfg.module.get("vision_backbone_type", "resnet18")
        if vis_backbone.startswith("vit"):
            VisModelClass = VisionViTPose
        else:
            VisModelClass = ResNetVisionPose
        vis_model = VisModelClass(
            out_channels=22,
            backbone_type=vis_backbone,
            pretrained=False,
            head_hidden=512,
            head_dropout=0.1,
        )
        vis_model.load_state_dict(vis_sd_remapped, strict=True)
        vis_model.to(device)
        vis_model.eval()
        print(f"Loaded standalone vision model: {sum(p.numel() for p in vis_model.parameters()):,} params")

    # ── Build dataset ──────────────────────────────────────────────────
    emg_layout = cfg.get("egoemg_emg_layout", "emg2pose_interpolate16")
    channel_indices = cfg.get("egoemg_emg2pose_channel_indices", [10, 12, 0, 1, 2, 4, 5, 6])
    channel_interpolate = cfg.get("egoemg_channel_interpolate", False)
    norm_stats_path = cfg.datamodule.get("per_dataset_norm_stats_path")
    val_window = cfg.datamodule.get("val_test_window_length", 7790)
    val_stride = cfg.datamodule.get("val_test_stride", val_window)
    crops_dir = cfg.get("per_episode_crops_dir")
    skip_emg = cfg.get("skip_emg_loading", False)

    all_results: list[dict] = []

    for split in args.splits:
        for hand in args.hands:
            print(f"\nEvaluating: split={split}, hand={hand}")

            dataset = EgoEmgMemmapDataset(
                memmap_dir=args.data_location,
                window_length=int(val_window),
                stride=int(val_stride),
                allowed_splits=[split],
                modalities=["emg", "joint_angles", "labels"],
                target_hand=hand,
                emg_field_preference="filtered",
                emg_layout=emg_layout,
                emg2pose_channel_indices=channel_indices,
                channel_interpolate=channel_interpolate,
                norm_mode="per-dataset",
                norm_stats_path=norm_stats_path,
                dataset_name="egoemg",
                jitter=False,
                per_episode_crops_dir=crops_dir,
                vision_num_frames=cfg.get("vision_num_frames", 0),
                vision_frame_selection=cfg.get("vision_frame_selection", "center"),
                vision_patch_size=cfg.get("vision_patch_size", 256),
                center_target_only=cfg.get("center_target_only", False),
                skip_emg_loading=skip_emg,
            )

            n_total = len(dataset)
            print(f"  Total samples: {n_total:,}")

            # 可选子采样
            if args.max_samples > 0 and n_total > args.max_samples:
                rng = np.random.default_rng(args.seed)
                indices = sorted(rng.choice(n_total, size=args.max_samples, replace=False).tolist())
            else:
                indices = list(range(n_total))

            # Custom dataset to carry abs_idx through to collate
            class _DatasetWithIdx(torch.utils.data.Dataset):
                def __init__(self, base, indices):
                    self.base = base
                    self.indices = indices

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, idx):
                    real_idx = self.indices[idx]
                    sample = self.base[real_idx]
                    sample["_abs_idx"] = real_idx
                    return sample

            def collate_fn(batch):
                from torch.utils.data._utils.collate import default_collate

                emg_batch = []
                for s in batch:
                    item = {}
                    for key in ["emg", "joint_angles", "label_valid_mask",
                                "vision_features", "vision_valid_mask", "vision_img",
                                "_abs_idx"]:
                        if key in s:
                            val = s[key]
                            if isinstance(val, np.ndarray):
                                val = torch.as_tensor(val, dtype=torch.float32 if key != "label_valid_mask" else torch.bool)
                            item[key] = val
                    emg_batch.append(item)
                return default_collate(emg_batch)

            ds_with_idx = _DatasetWithIdx(dataset, indices)
            dataloader = DataLoader(
                ds_with_idx, batch_size=args.batch_size, shuffle=False,
                num_workers=4, collate_fn=collate_fn, pin_memory=True,
            )

            split_results = evaluate_per_gesture(
                module, dataloader, dataset, device,
                emg_model=emg_model, vis_model=vis_model,
            )
            for r in split_results:
                r["split"] = split
                r["hand"] = hand
            all_results.extend(split_results)
            print(f"  Collected: {len(split_results)} samples")

    # ── Group by gesture ───────────────────────────────────────────────
    gesture_groups: dict[int, list[dict]] = defaultdict(list)
    for r in all_results:
        gesture_groups[r["gesture_class"]].append(r)

    # Sort by gesture class
    sorted_gestures = sorted(gesture_groups.keys())

    # ── Print & save results ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Per-Gesture Analysis — {len(all_results)} total samples, {len(sorted_gestures)} gestures")
    print(f"{'='*80}")
    header = f"{'Gesture':<10} {'N':<8} {'Vision(°)':<12} {'EMG(°)':<12} {'Fusion(°)':<12} {'Δ Vis-Fus(°)':<14} {'Δ EMG-Fus(°)':<14}"
    print(header)
    print("-" * 80)

    csv_rows = []
    for gc in sorted_gestures:
        items = gesture_groups[gc]
        n = len(items)
        vis_mae = np.mean([r["mae_vision"] for r in items]) * 57.3
        emg_mae = np.mean([r["mae_emg"] for r in items]) * 57.3
        fus_mae = np.mean([r["mae_fusion"] for r in items]) * 57.3
        delta_vf = vis_mae - fus_mae
        delta_ef = emg_mae - fus_mae
        label = f"G{gc}" if gc >= 0 else "invalid"
        print(f"{label:<10} {n:<8} {vis_mae:<12.2f} {emg_mae:<12.2f} {fus_mae:<12.2f} {delta_vf:<14.2f} {delta_ef:<14.2f}")
        csv_rows.append({
            "gesture_class": gc,
            "n": n,
            "vision_mae_deg": round(vis_mae, 2),
            "emg_mae_deg": round(emg_mae, 2),
            "fusion_mae_deg": round(fus_mae, 2),
            "delta_vis_fusion_deg": round(delta_vf, 2),
            "delta_emg_fusion_deg": round(delta_ef, 2),
        })

    # ── Overall summary ────────────────────────────────────────────────
    vis_all = np.mean([r["mae_vision"] for r in all_results]) * 57.3
    emg_all = np.mean([r["mae_emg"] for r in all_results]) * 57.3
    fus_all = np.mean([r["mae_fusion"] for r in all_results]) * 57.3
    print("-" * 80)
    print(f"{'OVERALL':<10} {len(all_results):<8} {vis_all:<12.2f} {emg_all:<12.2f} {fus_all:<12.2f} {vis_all-fus_all:<14.2f} {emg_all-fus_all:<14.2f}")

    # ── Save CSV ───────────────────────────────────────────────────────
    import pandas as pd
    csv_path = output_dir / "per_gesture_analysis.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Save summary JSON ──────────────────────────────────────────────
    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config_name,
        "total_samples": len(all_results),
        "n_gestures": len(sorted_gestures),
        "overall": {
            "vision_mae_deg": round(vis_all, 2),
            "emg_mae_deg": round(emg_all, 2),
            "fusion_mae_deg": round(fus_all, 2),
            "delta_vis_fusion_deg": round(vis_all - fus_all, 2),
            "delta_emg_fusion_deg": round(emg_all - fus_all, 2),
        },
        "per_gesture": csv_rows,
    }
    json_path = output_dir / "per_gesture_analysis.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
