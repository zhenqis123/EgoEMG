#!/usr/bin/env python
"""Unified center-frame evaluation for comparing fusion models trained with
different window lengths.

Problem: models trained with wl=7790 and wl=12000 produce val windows with
DIFFERENT center frame positions (stride=wl, so centers land on different
grids). Direct val_mae comparison is unfair — the two models are evaluated on
different sample sets.

Solution: this script picks a fixed set of center frames (the wl=7790 grid),
then evaluates each model on windows centered at those SAME frames. Each model
uses its own trained window_length, but the center (supervision target + vision
frame) is identical across models.

Usage:
  python scripts/eval/unified_center_eval.py
"""
from __future__ import annotations

import json
import sys
import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "config"

MODELS = {
    "egoemg_incre_showee_wl12000_bestaug_6gpu": {
        "ckpt": str(REPO) + "/logs/regression/egoemg_incre_showee_alltrain_wl12000_bestaug_6gpu/regression_egoemg_incre_showee_alltrain/version_0/checkpoints/egoemg-middle-wl12000-epoch=117-val_mae=0.2430.ckpt",
        "experiment": "emgformer/regression_egoemg_showee",
    },
}

# Reference WL for center grid: use 7790 (shorter → denser sampling)
REF_WL = 7790


def collect_val_centers(required_window_length: int | None = None):
    """Collect val centers on the reference grid.

    When ``required_window_length`` is supplied, retain only centers that can
    host that full window inside their episode.  This makes comparisons between
    models trained with different window lengths use an identical sample set.
    """
    import numpy as np

    memmap_dir = REPO / "data" / "EgoEMG_memmap"
    m = json.load(open(memmap_dir / "manifest.json"))
    N = m["total_rows"]
    ep = np.memmap(memmap_dir / "episode_index.dat", dtype=np.int64, mode="r", shape=(N,))
    fs = m["fields"]["frame_split_id"]
    split = np.memmap(
        memmap_dir / "frame_split_id.dat",
        dtype=fs["dtype"], mode="r", shape=tuple(fs["shape"]),
    )

    centers_per_episode = {}  # {ep_idx: [(center, start), ...]}
    for e in range(41):
        mask = ep[:] == e
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        s0, e_end = int(idx[0]), int(idx[-1])
        n = max(0, (e_end - s0 - REF_WL) // REF_WL + 1)
        if n == 0:
            continue
        starts = s0 + np.arange(n) * REF_WL
        centers = starts + REF_WL // 2
        center_splits = np.asarray(split[centers])
        val_mask = np.isin(center_splits, [1, 2, 3])  # user + gesture + both
        val_starts = starts[val_mask]
        val_centers = centers[val_mask]
        if required_window_length is not None:
            required_start = val_centers - required_window_length // 2
            required_end = required_start + required_window_length
            fits = (required_start >= s0) & (required_end <= e_end)
            val_starts = val_starts[fits]
            val_centers = val_centers[fits]
        if len(val_centers) > 0:
            centers_per_episode[e] = list(zip(val_centers.tolist(), val_starts.tolist()))

    total = sum(len(v) for v in centers_per_episode.values())
    print(
        f"Collected {total} val centers across {len(centers_per_episode)} episodes "
        f"(REF_WL={REF_WL}, required_window_length={required_window_length})"
    )
    return centers_per_episode


def _load_eval_config(info):
    """Load the exact training config when available, otherwise compose it."""
    if info.get("config_path"):
        cfg = OmegaConf.load(info["config_path"])
    else:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="base",
                overrides=[
                    f"experiment={info['experiment']}",
                    "train=false",
                    "eval=false",
                    "trainer.devices=[0]",
                    *info.get("config_overrides", []),
                ],
            )
    # A saved resolved config can be evaluated on another host where only data
    # roots differ.  Apply explicit dot-list overrides before interpolation is
    # resolved, preserving every training-time model and dataloader setting.
    if info.get("config_overrides"):
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(info["config_overrides"]))
    OmegaConf.resolve(cfg)
    # Lightning hparams saved by older experiments used *_conf names, whereas
    # the current train/eval entrypoints expect the Hydra group names below.
    for current_key, legacy_key in (
        ("module", "module_conf"),
        ("optimizer", "optimizer_conf"),
        ("lr_scheduler", "lr_scheduler_conf"),
    ):
        if current_key not in cfg and legacy_key in cfg:
            cfg[current_key] = cfg[legacy_key]

    # A full Lightning checkpoint contains both the vision and EMG branches.
    # Saved hparams can still point to transient pretraining checkpoints that
    # were used only to initialize training and may no longer be present.
    # Do not try to load those initializers before restoring the full model.
    if info.get("config_path"):
        for key in ("pretrained_checkpoint", "pretrained_emg_checkpoint"):
            if key in cfg:
                cfg[key] = None
        if "module" in cfg and "vision_pretrained_checkpoint" in cfg.module:
            cfg.module.vision_pretrained_checkpoint = None
    return cfg


def eval_model_on_centers(
    model_name,
    info,
    centers_per_episode,
    hand,
    predictions_dir: Path | None = None,
    emg_offset_samples: int = 0,
):
    """Evaluate a model on the unified center frames.

    For each center frame, construct a window of the MODEL's trained WL centered
    as close as possible to the reference center, then run forward and collect
    per-sample MAE.
    """ 
    import os

    cfg = _load_eval_config(info)
    ckpt_path = info["ckpt"]
    uses_vision = info.get("uses_vision", True)

    wl = int(cfg.datamodule.window_length)
    print(f"\n=== Evaluating {model_name} (wl={wl}) ===")

    # Build the dataset to get memmap readers.  Older saved fusion hparams
    # stored these fields only in datamodule.dataset_conf (rather than as the
    # current top-level egoemg_* aliases), so use the validation dataset entry
    # as a compatibility fallback.
    dataset_template = OmegaConf.select(cfg, "datamodule.dataset_conf.val.0", default={})

    def cfg_or_dataset(key, default=None):
        value = OmegaConf.select(cfg, key, default=None)
        if value is None:
            value = OmegaConf.select(dataset_template, key, default=default)
        return value

    channel_indices = cfg_or_dataset("egoemg_emg2pose_channel_indices")
    if channel_indices is None:
        # Legacy configs stored 1-based positions under this name.
        legacy_indices = OmegaConf.select(
            dataset_template, "emg2pose_channel_indices_1based", default=None
        )
        if legacy_indices is not None:
            channel_indices = [int(index) - 1 for index in legacy_indices]

    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

    # Create a dataset instance for data access (we'll call _getitem_center_supervised manually)
    ds = EgoEmgMemmapDataset(
        memmap_dir=cfg_or_dataset("egoemg_memmap_dir", default=dataset_template.memmap_dir),
        window_length=wl,
        stride=wl,  # doesn't matter, we bypass __getitem__
        allowed_splits=["user", "gesture", "both"],
        modalities=["emg", "joint_angles", "labels"],
        target_hand=hand,
        # The regenerated shared memmap retains the canonical paper filter;
        # legacy ``filtered`` arrays are deliberately not materialized there.
        # Use this one field for every model so cross-generation comparisons
        # have identical EMG inputs.
        emg_field_preference="filtered_paper",
        emg_layout=OmegaConf.select(
            cfg,
            "egoemg_emg_layout",
            default=OmegaConf.select(dataset_template, "emg_layout", default="target_hand"),
        ),
        emg2pose_channel_indices=channel_indices,
        channel_interpolate=bool(OmegaConf.select(
            cfg,
            "egoemg_channel_interpolate",
            default=OmegaConf.select(dataset_template, "channel_interpolate", default=False),
        )),
        norm_mode=cfg.datamodule.norm_mode,
        norm_stats_path=OmegaConf.select(
            cfg,
            "datamodule.per_dataset_norm_stats_path",
            default=OmegaConf.select(dataset_template, "norm_stats_path", default=None),
        ),
        dataset_name="egoemg",
        vision_num_frames=int(cfg_or_dataset("vision_num_frames", default=0)),
        per_episode_crops_dir=cfg_or_dataset("per_episode_crops_dir", default=None),
        vision_patch_size=int(
            cfg_or_dataset("vision_patch_size", default=256)
        ),
        # The shared center-frame reader emits both image and label fields.  It
        # reads EMG even for vision-only models, though the model ignores it.
        skip_emg_loading=False,
        center_target_only=bool(
            cfg_or_dataset("center_target_only", default=True)
        ),
    )

    # Load model
    from egoemg.train import make_lightning_module

    module = make_lightning_module(cfg)
    kwargs = {
        "module_conf": cfg.module,
        "optimizer_conf": cfg.optimizer,
        "lr_scheduler_conf": cfg.lr_scheduler,
        "loss_weights": cfg.loss_weights,
        "datamodule": cfg.get("datamodule"),
        # Override checkpoint hparams too: Lightning otherwise preserves the
        # original initialization-only paths from the saved checkpoint.
        "pretrained_checkpoint": None,
        "pretrained_emg_checkpoint": None,
        "stage2_vision_checkpoint": None,
        "map_location": "cpu",
    }
    # Trusted local Lightning checkpoints contain OmegaConf metadata.  Recent
    # Lightning/PyTorch defaults may otherwise request weights_only=True and
    # reject that metadata before loading the state dict.
    load_checkpoint = module.__class__.load_from_checkpoint
    if "weights_only" in inspect.signature(load_checkpoint).parameters:
        kwargs["weights_only"] = False
    module = load_checkpoint(ckpt_path, **kwargs)
    module.cuda().eval()
    model = module.model

    hand_idx = 0 if hand == "left" else 1
    hand_code = "L" if hand == "left" else "R"

    all_preds = []
    all_vision_preds = []
    all_deltas = []
    all_targets = []
    all_valid = []
    all_episode_indices = []
    all_centers = []
    all_split_ids = []
    all_subject_ids = []
    n_total = sum(len(v) for v in centers_per_episode.values())
    count = 0

    with torch.no_grad():
        for ep_idx, center_list in centers_per_episode.items():
            for ref_center, ref_start in center_list:
                # Adjust window to model's WL: center the model's window on ref_center
                model_start = ref_center - wl // 2
                model_end = model_start + wl

                # Check bounds.  Image and supervision stay centered at
                # ``ref_center``; only the EMG window may be shifted for the
                # temporal-alignment diagnostic.  A positive offset reads EMG
                # later than the image/label center, and a negative offset
                # reads earlier EMG.
                ep_start = int(ds._episode_start_idx[ep_idx])
                ep_end = int(ds._episode_end_idx[ep_idx])
                if model_start < ep_start or model_end > ep_end:
                    count += 1
                    continue  # window doesn't fit, skip
                emg_start = model_start + emg_offset_samples
                emg_end = model_end + emg_offset_samples
                if emg_start < ep_start or emg_end > ep_end:
                    count += 1
                    continue

                try:
                    if uses_vision:
                        sample = ds._getitem_center_supervised(
                            ep_idx, model_start, model_end, ref_center
                        )
                        if emg_offset_samples:
                            shifted = ds._getitem_center_supervised_emg(
                                ep_idx, emg_start, emg_end
                            )
                            sample["emg"] = shifted["emg"]
                    else:
                        sample = ds._getitem_center_supervised_emg(
                            ep_idx, emg_start, emg_end
                        )
                except Exception as exc:
                    if not all_preds:
                        print(f"  First sample error: {type(exc).__name__}: {exc}")
                    count += 1
                    continue

                # Collate to batch
                batch = {}
                for k, v in sample.items():
                    if isinstance(v, np.ndarray):
                        batch[k] = torch.from_numpy(v).float().cuda().unsqueeze(0)
                    elif isinstance(v, torch.Tensor):
                        batch[k] = v.cuda().unsqueeze(0) if v.ndim == 0 else v.cuda()
                    else:
                        batch[k] = v

                out = model(batch)
                if isinstance(out, tuple):
                    preds, targets, mask = out
                else:
                    preds = out
                    targets = batch.get("joint_angles")
                    mask = batch.get("label_valid_mask")

                sample_mask = batch.get("label_valid_mask")
                if sample_mask is not None:
                    valid = bool(sample_mask.reshape(-1)[0].cpu().item())
                else:
                    valid = True

                sample_targets = batch.get("joint_angles")
                if valid and sample_targets is not None:
                    if preds.ndim == 3:
                        preds = preds[:, :, preds.shape[-1] // 2]
                    p = preds.squeeze().cpu().numpy()
                    t = sample_targets.squeeze().cpu().numpy()
                    all_preds.append(p)
                    delta = getattr(model, "_last_delta", None)
                    if isinstance(delta, torch.Tensor) and delta.numel() == preds.numel():
                        delta_np = delta.squeeze().cpu().numpy()
                        all_deltas.append(delta_np)
                        all_vision_preds.append(p - delta_np)
                    all_targets.append(t)
                    all_valid.append(True)
                    all_episode_indices.append(ep_idx)
                    all_centers.append(ref_center)
                    all_split_ids.append(
                        int(ds._frame_memmaps["frame_split_id"][ref_center])
                    )
                    all_subject_ids.append(int(ds._episode_subject_id[ep_idx]))

                count += 1
                if count % 200 == 0:
                    print(f"  {count}/{n_total}...")

    preds = np.array(all_preds)  # (N, 22)
    targets = np.array(all_targets)  # (N, 22)
    n_valid = len(preds)
    print(f"  Evaluated {n_valid}/{n_total} valid samples")
    if n_valid == 0:
        raise RuntimeError(f"No valid samples evaluated for {model_name}/{hand}")

    # Per-joint MAE
    mae_per_joint = np.abs(preds - targets).mean(axis=0)  # (22,)
    overall_mae = mae_per_joint.mean()

    residual_diagnostic = None
    if len(all_vision_preds) == n_valid:
        vision_preds = np.asarray(all_vision_preds)
        deltas = np.asarray(all_deltas)
        vision_overall_mae = np.abs(vision_preds - targets).mean(axis=0).mean()
        delta_abs_mean = np.abs(deltas).mean()
        print(
            f"  Residual diagnostic: vision_base_mae={vision_overall_mae:.4f}, "
            f"mean_abs_delta={delta_abs_mean:.4f}"
        )
        residual_diagnostic = {
            "vision_base_mae": float(vision_overall_mae),
            "mean_abs_delta": float(delta_abs_mean),
        }

    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in model_name
        )
        prediction_payload = {
            "predictions": preds,
            "targets": targets,
            "episode_indices": np.asarray(all_episode_indices, dtype=np.int64),
            "centers": np.asarray(all_centers, dtype=np.int64),
            "split_ids": np.asarray(all_split_ids, dtype=np.int64),
            "subject_ids": np.asarray(all_subject_ids, dtype=np.int64),
        }
        if len(all_vision_preds) == n_valid:
            prediction_payload["vision_predictions"] = np.asarray(all_vision_preds)
        np.savez_compressed(
            predictions_dir / f"{safe_name}_{hand}.npz",
            **prediction_payload,
        )

    # Per-finger groups
    finger_groups = {
        "thumb": [0, 1, 2, 3],
        "index": [4, 5, 6, 7],
        "middle": [8, 9, 10, 11],
        "ring": [12, 13, 14, 15],
        "pinky": [16, 17, 18, 19],
        "wrist": [20, 21],
    }

    print(f"  Overall MAE: {overall_mae:.4f}")
    for fname, jidx in finger_groups.items():
        fmae = mae_per_joint[jidx].mean()
        print(f"    {fname}: {fmae:.4f}")

    result = {"overall": overall_mae, "per_joint": mae_per_joint.tolist(), "n": n_valid}
    if residual_diagnostic is not None:
        result["residual_diagnostic"] = residual_diagnostic
    return result


def eval_model_both_hands(
    model_name,
    info,
    centers_per_episode,
    predictions_dir: Path | None = None,
    emg_offset_samples: int = 0,
):
    """Evaluate left and right hands, then combine by valid element count."""
    by_hand = {}
    for hand in ("left", "right"):
        by_hand[hand] = eval_model_on_centers(
            model_name,
            info,
            centers_per_episode,
            hand=hand,
            predictions_dir=predictions_dir,
            emg_offset_samples=emg_offset_samples,
        )

    n_total = sum(result["n"] for result in by_hand.values())
    if n_total == 0:
        raise RuntimeError(f"No valid samples for either hand: {model_name}")
    combined_per_joint = sum(
        np.asarray(result["per_joint"], dtype=np.float64) * result["n"]
        for result in by_hand.values()
    ) / n_total
    combined = {
        "overall": float(combined_per_joint.mean()),
        "per_joint": combined_per_joint.tolist(),
        "n": n_total,
    }
    print(
        f"  Both hands: MAE = {combined['overall']:.4f} "
        f"({combined['n']} hand samples; "
        f"left={by_hand['left']['overall']:.4f}, "
        f"right={by_hand['right']['overall']:.4f})"
    )
    return {**by_hand, "combined": combined}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-json",
        type=Path,
        help="JSON mapping of model name to ckpt/experiment/uses_vision metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for machine-readable evaluation results.",
    )
    parser.add_argument(
        "--model-names",
        help="Comma-separated subset of keys in --models-json.",
    )
    parser.add_argument(
        "--center-window-length",
        type=int,
        help="Keep only centers that can host this full window in every episode.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Optional directory for aligned per-sample prediction NPZ files.",
    )
    parser.add_argument(
        "--emg-offset-samples",
        type=int,
        default=0,
        help=(
            "Shift only the EMG window relative to the fixed image/label center. "
            "Positive values read later EMG; negative values read earlier EMG."
        ),
    )
    args = parser.parse_args()

    models = MODELS
    if args.models_json is not None:
        with args.models_json.open() as f:
            models = json.load(f)
    if args.model_names:
        requested = [name for name in args.model_names.split(",") if name]
        unknown = sorted(set(requested) - set(models))
        if unknown:
            raise ValueError(f"Unknown model names: {unknown}")
        models = {name: models[name] for name in requested}
    print("=" * 70)
    print("Unified Center-Frame Evaluation")
    print(f"Reference grid: WL={REF_WL}, stride={REF_WL} (no overlap, no jitter)")
    print("=" * 70)

    centers_per_episode = collect_val_centers(args.center_window_length)

    results = {}
    for model_name, info in models.items():
        if not Path(info["ckpt"]).exists():
            print(f"\n Skipping {model_name}: checkpoint not found")
            continue
        r = eval_model_both_hands(
            model_name,
            info,
            centers_per_episode,
            predictions_dir=args.predictions_dir,
            emg_offset_samples=args.emg_offset_samples,
        )
        results[model_name] = r

    # Summary comparison
    print("\n" + "=" * 70)
    print("UNIFIED COMPARISON (same center frames)")
    print("=" * 70)
    for name, r in results.items():
        combined = r["combined"]
        print(
            f"  {name}: combined={combined['overall']:.4f} "
            f"left={r['left']['overall']:.4f} right={r['right']['overall']:.4f} "
            f"({combined['n']} hand samples)"
        )
    if len(results) == 2:
        names = list(results.keys())
        diff = (
            results[names[0]]["combined"]["overall"]
            - results[names[1]]["combined"]["overall"]
        )
        print(f"\n  Difference ({names[0]} - {names[1]}): {diff:+.4f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(
                results,
                f,
                indent=2,
                default=lambda value: value.item()
                if isinstance(value, np.generic)
                else TypeError(f"Not JSON serializable: {type(value).__name__}"),
            )
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
