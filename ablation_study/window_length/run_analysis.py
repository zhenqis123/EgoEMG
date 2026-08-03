#!/usr/bin/env python3
"""Run per-timestep MAE analysis for all window-length trials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from egoemg.lightning import EmgPredictionModule

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRIALS = [
    {
        "study": "short-sweep",
        "trial": 0,
        "wl": 1000,
        "val_mae": 0.2853,
        "trial_dir": "logs/optuna_window/short_window_sweep/wl_1000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=001-val_mae=0.2853.ckpt",
    },
    {
        "study": "short-sweep",
        "trial": 1,
        "wl": 3000,
        "val_mae": 0.2750,
        "trial_dir": "logs/optuna_window/short_window_sweep/wl_3000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=024-val_mae=0.2750.ckpt",
    },
    {
        "study": "short-sweep",
        "trial": 2,
        "wl": 5000,
        "val_mae": 0.2704,
        "trial_dir": "logs/optuna_window/short_window_sweep/wl_5000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=066-val_mae=0.2704.ckpt",
    },
    {
        "study": "short-sweep",
        "trial": 3,
        "wl": 7000,
        "val_mae": 0.2652,
        "trial_dir": "logs/optuna_window/short_window_sweep/wl_7000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=137-val_mae=0.2652.ckpt",
    },
    {
        "study": "short-sweep",
        "trial": 4,
        "wl": 9000,
        "val_mae": 0.2618,
        "trial_dir": "logs/optuna_window/short_window_sweep/wl_9000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=049-val_mae=0.2618.ckpt",
    },
    {
        "study": "short-sweep",
        "trial": 5,
        "wl": 11000,
        "val_mae": 0.2495,
        "trial_dir": "logs/optuna_window/short_window_sweep/wl_11000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=132-val_mae=0.2495.ckpt",
    },
    {
        "study": "short-sweep",
        "trial": 6,
        "wl": 13000,
        "val_mae": 0.2490,
        "trial_dir": "logs/optuna_window/short_window_sweep/wl_13000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=133-val_mae=0.2490.ckpt",
    },
    {
        "study": "egoemg-window-v3",
        "trial": 6,
        "wl": 7494,
        "val_mae": 0.2586,
        "trial_dir": "logs/optuna_window/egoemg-window-v3/trial_0006_2026-05-20_22-16-15",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=088-val_mae=0.2586.ckpt",
    },
    {
        "study": "egoemg-window-v3",
        "trial": 7,
        "wl": 14409,
        "val_mae": 0.2428,
        "trial_dir": "logs/optuna_window/egoemg-window-v3/trial_0007_2026-05-20_23-48-05",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=059-val_mae=0.2428.ckpt",
    },
    {
        "study": "egoemg-window-v4",
        "trial": 6,
        "wl": 14638,
        "val_mae": 0.2401,
        "trial_dir": "logs/optuna_window/egoemg-window-v4/trial_0006_2026-05-21_11-25-39",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=133-val_mae=0.2401.ckpt",
    },
    {
        "study": "egoemg-window-v4",
        "trial": 5,
        "wl": 15716,
        "val_mae": 0.2330,
        "trial_dir": "logs/optuna_window/egoemg-window-v4/trial_0005_2026-05-21_10-00-47",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=131-val_mae=0.2330.ckpt",
    },
    {
        "study": "egoemg-window-v4",
        "trial": 0,
        "wl": 18120,
        "val_mae": 0.2255,
        "trial_dir": "logs/optuna_window/egoemg-window-v4/trial_0000_2026-05-21_02-56-08",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=134-val_mae=0.2255.ckpt",
    },
    {
        "study": "egoemg-window-v4",
        "trial": 3,
        "wl": 20585,
        "val_mae": 0.2177,
        "trial_dir": "logs/optuna_window/egoemg-window-v4/trial_0003_2026-05-21_07-07-59",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=124-val_mae=0.2177.ckpt",
    },
    {
        "study": "egoemg-window-v4",
        "trial": 2,
        "wl": 22052,
        "val_mae": 0.2156,
        "trial_dir": "logs/optuna_window/egoemg-window-v4/trial_0002_2026-05-21_05-43-23",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=123-val_mae=0.2156.ckpt",
    },
    {
        "study": "egoemg-window-v4",
        "trial": 1,
        "wl": 24458,
        "val_mae": 0.2090,
        "trial_dir": "logs/optuna_window/egoemg-window-v4/trial_0001_2026-05-21_04-20-31",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=130-val_mae=0.2090.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 0,
        "wl": 25000,
        "val_mae": 0.2033,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_25000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=133-val_mae=0.2033.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 1,
        "wl": 26000,
        "val_mae": 0.2018,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_26000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=129-val_mae=0.2018.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 2,
        "wl": 27000,
        "val_mae": 0.2057,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_27000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=138-val_mae=0.2057.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 3,
        "wl": 28000,
        "val_mae": 0.1977,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_28000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=137-val_mae=0.1977.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 4,
        "wl": 29000,
        "val_mae": 0.1927,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_29000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=120-val_mae=0.1927.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 5,
        "wl": 30000,
        "val_mae": 0.1927,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_30000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=142-val_mae=0.1927.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 6,
        "wl": 31000,
        "val_mae": 0.1925,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_31000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=122-val_mae=0.1925.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 7,
        "wl": 32000,
        "val_mae": 0.1914,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_32000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=126-val_mae=0.1914.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 8,
        "wl": 33000,
        "val_mae": 0.1912,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_33000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=119-val_mae=0.1912.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 9,
        "wl": 34000,
        "val_mae": 0.1892,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_34000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=116-val_mae=0.1892.ckpt",
    },
    {
        "study": "saturation-sweep",
        "trial": 10,
        "wl": 35000,
        "val_mae": 0.1858,
        "trial_dir": "logs/optuna_window/saturation_sweep/wl_35000",
        "ckpt": "regression_emgformer_middle_aug_search_egoemg/version_0/checkpoints/egoemg-middle-epoch=142-val_mae=0.1858.ckpt",
    },
]


def analyze_trial(trial_info: dict) -> dict:
    trial_dir = PROJECT_ROOT / trial_info["trial_dir"]
    ckpt_path = trial_dir / trial_info["ckpt"]
    config_path = trial_dir / "hydra_configs" / "config.yaml"

    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    cfg = OmegaConf.load(config_path)
    module = EmgPredictionModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    module.eval()
    module.cuda()

    val_datasets = [instantiate(ds_cfg) for ds_cfg in cfg.dataset.val]
    val_dataset = ConcatDataset(val_datasets)
    loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    T_out = None
    timestep_sum = None
    timestep_count = None

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  wl={trial_info['wl']}", leave=False):
            batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            preds, targets, mask = module(batch_cuda)

            abs_err = (preds - targets).abs()
            if mask.ndim == 2:
                mask_expanded = mask.unsqueeze(1).expand_as(abs_err)
            else:
                mask_expanded = mask

            abs_err = abs_err * mask_expanded
            T = abs_err.shape[-1]

            if T_out is None:
                T_out = T
                timestep_sum = torch.zeros(T, device="cuda")
                timestep_count = torch.zeros(T, device="cuda")

            timestep_sum += abs_err.sum(dim=(0, 1))
            timestep_count += mask_expanded.sum(dim=(0, 1))

    timestep_mae = (timestep_sum / timestep_count.clamp(min=1)).cpu().numpy()

    out_path = RESULTS_DIR / f"wl_{trial_info['wl']}.npy"
    np.save(out_path, timestep_mae)

    del module
    torch.cuda.empty_cache()

    return {
        "wl": trial_info["wl"],
        "val_mae": trial_info["val_mae"],
        "study": trial_info["study"],
        "trial": trial_info["trial"],
        "T_out": int(T_out),
        "overall_mae": float(timestep_mae.mean()),
        "min_mae": float(timestep_mae.min()),
        "max_mae": float(timestep_mae.max()),
        "min_mae_pos": float(timestep_mae.argmin() / T_out),
        "max_mae_pos": float(timestep_mae.argmax() / T_out),
        "npy_file": f"wl_{trial_info['wl']}.npy",
    }


def main():
    print(f"Analyzing {len(TRIALS)} trials...")
    results = []

    for i, trial_info in enumerate(TRIALS):
        print(f"[{i+1}/{len(TRIALS)}] Study={trial_info['study']}, wl={trial_info['wl']}")
        result = analyze_trial(trial_info)
        if result is not None:
            results.append(result)
            print(f"  -> overall_mae={result['overall_mae']:.4f}, T={result['T_out']}")

    summary = {
        "experiment": "Window Length Ablation",
        "model": "EMGFormer Middle (6.6M params)",
        "dataset": "EgoEMG (val split: user+gesture+both)",
        "search_ranges": ["v3: 3000-15000", "v4: 14000-25000"],
        "total_trials_analyzed": len(results),
        "trials": sorted(results, key=lambda x: x["wl"]),
    }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_DIR}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
