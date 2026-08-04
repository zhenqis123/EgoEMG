# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Offline evaluation for trained EMG2Pose models using memmap datasets."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from egoemg.constants import FINGERS, JOINTS, PD_GROUPS
from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
from egoemg.lightning import EmgPredictionModule
from egoemg.train import _extract_state_dict, _is_pretrain_checkpoint, make_lightning_module
from egoemg.transforms import Compose



def _results_output_path() -> str:
    """Where to write the results.csv eval artifact.

    Uses the Hydra run output dir when available, so eval runs never clobber
    a results.csv in the working directory (repo root). Falls back to cwd.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        output_dir = os.getcwd()
    return os.path.abspath(os.path.join(output_dir, "results.csv"))


class _DatasetWithIndex(torch.utils.data.Dataset):
    """Thin wrapper that adds the dataset index and metadata to each sample."""

    def __init__(self, base: torch.utils.data.Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        sample["_idx"] = idx
        # Carry through user/session metadata if present (Emg2PoseDataset puts
        # "user" in each sample). The default collate produces a tuple of strings.
        if "user" in sample:
            sample["_user"] = sample["user"]
        return sample


def infer_dataset_type(config: DictConfig) -> str:
    """Infer evaluation dataset family from Hydra dataset config."""
    dataset_cfg = config.get("dataset")
    if dataset_cfg is None:
        return "emg2pose"

    def _iter_cfgs(value: Any) -> list[Any]:
        if value is None:
            return []
        if OmegaConf.is_list(value):
            return list(value)
        if OmegaConf.is_dict(value):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    for split_name in ("train", "val", "test"):
        for cfg in _iter_cfgs(dataset_cfg.get(split_name)):
            target = ""
            if OmegaConf.is_dict(cfg) or isinstance(cfg, dict):
                target = str(cfg.get("_target_", "") or "")
            if "egoemg_memmap_dataset" in target.lower():
                return "egoemg"

    if config.get("egoemg_unified_memmap_dir") or config.get("egoemg_memmap_dir"):
        return "egoemg"
    return "emg2pose"


@dataclass
class EMG2PoseEvaluation:
    """
    Run offline evaluation for a trained emg2pose model using memmap datasets.

    Metrics are computed for each combination of conditions, e.g. for each
    (generalization, user) or each (generalization, user, stage).
    """

    config: DictConfig
    checkpoint: str
    conditions: list[str] = field(default_factory=lambda: ["generalization"])
    window_length: int = 10_000
    split: str = "test"
    skip_ik_failures: bool = True
    batch_size: int = 512
    dataset_type: str = "emg2pose"  # "emg2pose" or "egoemg"
    norm_mode_override: str | None = None  # override norm_mode for experiments
    per_user: bool = False
    per_group_stats: bool = False  # compute per-user & per-gesture mean±std (EgoEMG only)

    def __post_init__(self):
        self.data_dir = self.config.data_location
        self.batch_size = self.config.get("batch_size", self.batch_size)
        self.num_workers = self.config.get("num_workers", 12)
        inferred_dataset_type = infer_dataset_type(self.config)
        if self.dataset_type == "emg2pose" and inferred_dataset_type != "emg2pose":
            # Only auto-switch if the experiment config doesn't carry egoemg
            # settings that are irrelevant to the emg2pose evaluation dataset.
            # The user may explicitly want to evaluate a pretrain/egoemg model
            # on the emg2pose benchmark.
            if not (self.config.get("egoemg_unified_memmap_dir") or self.config.get("egoemg_memmap_dir")):
                self.dataset_type = inferred_dataset_type

        # Determine memmap directory
        self.memmap_dir = self._find_memmap_dir()

        self.is_pretrain = False  # set by get_module()
        self.module = self.get_module()

        if self.dataset_type == "egoemg":
            self.dataloaders = self._get_egoemg_dataloaders()
        else:
            self.df = self.get_corpus_df()
            self.groupby = self.df.groupby(self.conditions)
            self.dataloaders = self.get_dataloaders()

    def _find_memmap_dir(self) -> str:
        """Find memmap directory from the unified/legacy config keys, then data_location."""
        for key in ("egoemg_unified_memmap_dir", "egoemg_memmap_dir"):
            memmap = self.config.get(key)
            if memmap and os.path.exists(os.path.join(str(memmap).rstrip("/"), "manifest.json")):
                return str(memmap)
        data_dir = self.data_dir.rstrip("/")
        # Check if data_location is already memmap
        if os.path.exists(os.path.join(data_dir, "manifest.json")):
            return data_dir
        # Try _memmap suffix
        memmap_dir = data_dir + "_memmap"
        if os.path.exists(memmap_dir):
            return memmap_dir
        # Fallback to original data_dir (might be zarr)
        return data_dir

    def get_corpus_df(self) -> pd.DataFrame:
        """Load metadata and filter by split."""
        # Try loading from metadata.npz (memmap format)
        possible_npz_paths = [
            os.path.join(self.data_dir, "metadata.npz"),
            os.path.join(self.data_dir.rstrip("/") + "_memmap", "metadata.npz"),
        ]

        for npz_path in possible_npz_paths:
            if os.path.exists(npz_path):
                return self._get_corpus_df_from_npz(npz_path)

        # Fallback to metadata.csv
        metadata_file = os.path.join(self.data_dir, "metadata.csv")
        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file).query(f"split=='{self.split}'")
            corpus_subsample = self.config.get("corpus_subsample", 1)
            if corpus_subsample < 1:
                print(f"Subsampling corpus by {corpus_subsample}")
                df = df.sample(frac=corpus_subsample, random_state=0)
            return df

        raise FileNotFoundError(
            f"No metadata found. Tried: {possible_npz_paths} and {metadata_file}"
        )

    def _get_corpus_df_from_npz(self, npz_path: str) -> pd.DataFrame:
        """Load corpus dataframe from metadata.npz."""
        data = np.load(npz_path)

        def decode(arr):
            return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]

        records = []
        n_sessions = len(data["session_start_idx"])

        users = decode(data["users_user"])
        stages = decode(data.get("stages_stage", []))
        splits = decode(data.get("splits_split", ["train", "val", "test"]))

        for i in range(n_sessions):
            record = {
                "session_id": decode([data["session_session_id"][i]])[0],
                "filename": decode([data["session_filename"][i]])[0] if "session_filename" in data else decode([data["session_session_id"][i]])[0],
                "user": users[int(data["session_user_id"][i])],
                "start_idx": int(data["session_start_idx"][i]),
                "end_idx": int(data["session_end_idx"][i]),
                "length": int(data["session_length"][i]),
            }

            if "session_stage_id" in data and len(stages) > 0:
                stage_id = int(data["session_stage_id"][i])
                record["stage"] = stages[stage_id] if stage_id < len(stages) else "unknown"

            if "session_split_id" in data:
                split_id = int(data["session_split_id"][i])
                record["split"] = splits[split_id] if split_id < len(splits) else "unknown"
            else:
                record["split"] = self.split

            if "session_generalization" in data:
                record["generalization"] = decode([data["session_generalization"][i]])[0]
            else:
                if data.get("session_held_out_user", [False]*n_sessions)[i]:
                    record["generalization"] = "cross_user"
                elif data.get("session_held_out_stage", [False]*n_sessions)[i]:
                    record["generalization"] = "cross_stage"
                else:
                    record["generalization"] = "seen_user"

            records.append(record)

        df = pd.DataFrame(records)

        if "split" in df.columns:
            df = df.query(f"split=='{self.split}'")

        corpus_subsample = self.config.get("corpus_subsample", 1)
        if corpus_subsample < 1:
            print(f"Subsampling corpus by {corpus_subsample}")
            df = df.sample(frac=corpus_subsample, random_state=0)

        return df

    def get_module(self):
        """Load the Lightning module from checkpoint (supports pretrain checkpoints)."""
        ckpt_path = self.checkpoint or self.config.get("checkpoint")
        if not ckpt_path:
            print("Warning: No checkpoint provided, returning None for module")
            return None

        ckpt_path = os.path.abspath(ckpt_path)

        # Detect if this is a pretrain checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        is_pretrain = _is_pretrain_checkpoint(state_dict)

        if is_pretrain:
            self.is_pretrain = True
            print(f"Detected pretrain checkpoint: {ckpt_path}")
            print("Loading backbone + angle_head into regression module.")
            # Bypass make_lightning_module which rejects pretrain configs;
            # create EmgPredictionModule directly for evaluation.
            module = EmgPredictionModule(
                module_conf=self.config.module,
                optimizer_conf=self.config.optimizer,
                lr_scheduler_conf=self.config.lr_scheduler,
                loss_weights=self.config.loss_weights,
                pretrained_checkpoint=self.config.get("pretrained_checkpoint"),
                pretrained_strict=self.config.get("pretrained_strict", False),
                freeze_backbone=self.config.get("freeze_backbone", False),
                ignore_head_tail_dims=self.config.get("ignore_head_tail_dims", 0),
                datamodule=self.config.get("datamodule"),
            )
            module._load_pretrained_backbone(ckpt_path, strict=False)
            module._load_pretrained_angle_head(ckpt_path, strict=False)
        else:
            print(f"Detected regular checkpoint: {ckpt_path}")
            model_target = self.config.module._target_
            map_location = "cpu"

            module = make_lightning_module(self.config)
            kwargs = dict(
                module_conf=self.config.module,
                optimizer_conf=self.config.optimizer,
                lr_scheduler_conf=self.config.lr_scheduler,
                loss_weights=self.config.loss_weights,
                map_location=map_location,
            )
            if hasattr(self.config, 'datamodule') and self.config.datamodule is not None:
                kwargs['datamodule'] = self.config.datamodule
            # Checkpoints bake their training-time init paths (pretrained EMG
            # backbone, vision checkpoint) into hparams. Those absolute paths
            # may not exist on the evaluating machine; for evaluation the
            # trained weights come from this checkpoint, so skip init loads
            # instead of crashing (F3: portability of released checkpoints).
            ckpt_hparams = (
                checkpoint.get("hyper_parameters", {})
                if isinstance(checkpoint, dict)
                else {}
            )
            for key in (
                "pretrained_emg_checkpoint",
                "pretrained_checkpoint",
                "stage2_vision_checkpoint",
                "vision_pretrained_checkpoint",
            ):
                hp_val = ckpt_hparams.get(key)
                if hp_val and not os.path.exists(os.path.expanduser(str(hp_val))):
                    kwargs[key] = None
                    print(
                        f"Note: {key}={hp_val} does not exist on this machine; "
                        "skipping pretrained-init load during evaluation"
                    )
            module = module.__class__.load_from_checkpoint(ckpt_path, **kwargs)

        module.eval()
        return module

    def _get_egoemg_dataloaders(self) -> list[DataLoader]:
        """Create EgoEMG test dataloaders grouped by generalization split."""
        from egoemg.datasets.pretrain_wrapper import PretrainWrapperDataset

        memmap_dir = self.config.get("egoemg_unified_memmap_dir") or self.config.get("egoemg_memmap_dir", "")
        emg_layout = self.config.get("egoemg_emg_layout", "emg2pose_interpolate16")
        channel_indices = self.config.get("egoemg_emg2pose_channel_indices",
                                          [10, 12, 0, 1, 2, 4, 5, 6])
        channel_interpolate = self.config.get("egoemg_channel_interpolate", True)
        norm_stats_path = self.config.datamodule.get("per_dataset_norm_stats_path", None)

        # Allow config to override which splits to evaluate.
        # Default: ["user", "gesture", "both"] for test-time generalization analysis.
        # Override to ["val"] to match pretrain-time validation conditions.
        config_splits = self.config.get("egoemg_test_splits", None)
        splits = config_splits if config_splits else ["user", "gesture", "both"]

        # Allow config to override group names for results reporting
        config_group_names = self.config.get("egoemg_group_names", None)

        # Use window_length and stride from datamodule config if available
        dm_window = self.config.datamodule.get("val_test_window_length", None) if hasattr(self.config, 'datamodule') else None
        dm_stride = self.config.datamodule.get("val_test_stride", None) if hasattr(self.config, 'datamodule') else None
        base_window = int(dm_window) if dm_window else self.window_length
        stride = int(dm_stride) if dm_stride else base_window

        dataloaders = []
        group_names = []
        wrapped_datasets = []  # stored for per-group evaluation

        for split_name in splits:
            for hand in ["left", "right"]:
                dataset = PretrainWrapperDataset(
                    base={
                        '_target_': 'egoemg.datasets.egoemg_memmap_dataset.EgoEmgMemmapDataset',
                        'memmap_dir': memmap_dir,
                        # Use the configured evaluation window directly. The model
                        # already accounts for its temporal context when aligning
                        # predictions and targets inside the Lightning module.
                        'window_length': base_window,
                        'stride': stride,
                        'allowed_splits': [split_name],
                        'modalities': ['emg', 'joint_angles', 'labels'],
                        'target_hand': hand,
                        'emg_field_preference': self.config.get('egoemg_emg_field_preference', 'filtered'),
                        'emg_layout': emg_layout,
                        'emg2pose_channel_indices': channel_indices,
                        'channel_interpolate': channel_interpolate,
                        # Let the base dataset handle normalization so that
                        # field-aware / per-channel stats are used correctly.
                        # PretrainWrapper norm is disabled to avoid mismatch.
                        'norm_mode': 'per-dataset',
                        'norm_stats_path': norm_stats_path,
                    },
                    name='egoemg',
                    angle_dim=22,
                    norm_mode=None,  # base dataset already normalizes
                    norm_stats_path=None,
                )
                if len(dataset) == 0:
                    continue

                wrapped_datasets.append(dataset)

                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    prefetch_factor=2 if self.num_workers > 0 else None,
                    persistent_workers=self.num_workers > 0,
                    collate_fn=_egoemg_collate_fn,
                )
                dataloaders.append(dataloader)
                if config_group_names:
                    group_names.append(config_group_names[len(group_names)])
                else:
                    group_names.append(f"{split_name}/{hand}")
        self._egoemg_group_names = group_names
        self._egoemg_wrapped_datasets = wrapped_datasets
        return dataloaders

    def get_dataloaders(self) -> list[DataLoader]:
        """Get list of dataloaders for emg2pose dataset."""
        if hasattr(self.config, 'transforms') and self.config.transforms is not None:
            if self.split in self.config.transforms and self.config.transforms[self.split] is not None:
                transforms = Compose(instantiate(self.config.transforms[self.split], _convert_="all"))
            else:
                from egoemg.transforms import ExtractStructuredFields
                transforms = ExtractStructuredFields()
        else:
            from egoemg.transforms import ExtractStructuredFields
            transforms = ExtractStructuredFields()

        print("Creating dataloaders for each condition.")
        dataloaders = []
        self._emg2pose_datasets = []  # stored for per-user evaluation
        for _, df_ in tqdm(self.groupby):
            session_ids = df_["session_id"].tolist() if "session_id" in df_.columns else df_["filename"].tolist()

            ds_kwargs = dict(
                memmap_dir=self.memmap_dir,
                # Match the training/test datamodule windowing exactly; model
                # context is handled internally during forward/alignment.
                window_length=self.window_length,
                stride=self.window_length,
                padding=(0, 0),
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
                allowed_sessions=session_ids,
                transform=transforms,
            )

            # Apply normalization settings from the datamodule config to match
            # training-time preprocessing (not just for pretrain checkpoints).
            dm_cfg = self.config.get("datamodule")
            if dm_cfg is not None:
                norm_mode = dm_cfg.get("norm_mode")
                if norm_mode and norm_mode != "batch":
                    ds_kwargs["norm_mode"] = norm_mode
                    if norm_mode == "per-dataset":
                        ds_kwargs["norm_stats_path"] = dm_cfg.get(
                            "per_dataset_norm_stats_path"
                        )
                        ds_kwargs["dataset_name"] = "emg2pose"

            dataset = Emg2PoseDataset(**ds_kwargs)
            self._emg2pose_datasets.append(dataset)

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=2 if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
            )
            dataloaders.append(dataloader)

        return dataloaders

    def _evaluate_egoemg_per_group(self) -> list[dict]:
        """Manual evaluation pass collecting per-user & per-gesture mean±std.

        Returns a list of dicts (one per dataloader) with keys:
        per_user_mae_mean, per_user_mae_std, per_user_count,
        per_gesture_mae_mean, per_gesture_mae_std, per_gesture_count.

        MAE is accumulated incrementally on CPU to avoid OOM.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = self.module
        module.to(device)
        module.eval()

        # Pre-build per-group channel index arrays for per-user per-joint stats
        group_indices: dict[str, list[int]] = {}
        for finger in FINGERS:
            group_indices[finger] = [j.index for j in JOINTS if finger in j.groups]
        for pd_group in PD_GROUPS:
            group_indices[pd_group] = [j.index for j in JOINTS if pd_group in j.groups]
        # Wrist articulation channels (indices 20, 21; not in JOINTS)
        group_indices["wrist_flexion"] = [20]
        group_indices["wrist_deviation"] = [21]

        all_stats: list[dict] = []

        for ds_idx, wrapper_ds in enumerate(
            tqdm(self._egoemg_wrapped_datasets, desc="Per-group")
        ):
            base_ds = wrapper_ds._base
            n = len(wrapper_ds)

            # Build index metadata: (subject, gesture_class) per sample
            index_meta: list[tuple[str, int]] = [
                (base_ds.get_subject(i), base_ds.get_gesture_class(i))
                for i in range(n)
            ]

            indexed_ds = _DatasetWithIndex(wrapper_ds)
            loader = DataLoader(
                indexed_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                collate_fn=_egoemg_collate_fn,
            )

            user_sum_abs: dict[str, float] = {}
            user_count: dict[str, int] = {}
            gesture_sum_abs: dict[str, float] = {}
            gesture_count: dict[str, int] = {}

            # Per-user per-group accumulators
            user_group_sum_abs: dict[str, dict[str, float]] = {}
            user_group_count: dict[str, dict[str, int]] = {}

            for batch in loader:
                indices = batch.pop("_idx", None)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with torch.no_grad():
                    preds, targets, mask = module.forward(batch)

                # Vectorized per-sample L1 on GPU
                C = preds.shape[1]
                abs_per_time = (preds - targets).abs().sum(dim=1)  # (B, T)
                sample_abs = (abs_per_time * mask.float()).sum(dim=1)  # (B,)
                sample_n = mask.float().sum(dim=1) * C  # (B,)
                sa = sample_abs.cpu().numpy()
                sn = sample_n.cpu().numpy()

                # Per-group per-sample MAE (vectorized, one kernel per group)
                g_sa: dict[str, np.ndarray] = {}
                g_sn: dict[str, np.ndarray] = {}
                for gname, idxs in group_indices.items():
                    gC = len(idxs)
                    idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
                    g_abs_per_time = (preds[:, idxs_t] - targets[:, idxs_t]).abs().sum(dim=1)  # (B, T)
                    g_sample_abs = (g_abs_per_time * mask.float()).sum(dim=1)  # (B,)
                    g_sample_n = mask.float().sum(dim=1) * gC  # (B,)
                    g_sa[gname] = g_sample_abs.cpu().numpy()
                    g_sn[gname] = g_sample_n.cpu().numpy()

                # Accumulate per user/gesture (CPU-side)
                batch_indices = indices.cpu().numpy() if indices is not None else range(len(sa))
                for j in range(len(sa)):
                    if sn[j] == 0:
                        continue
                    idx = int(batch_indices[j])
                    subject, gesture = index_meta[idx]

                    user_sum_abs[subject] = user_sum_abs.get(subject, 0.0) + float(sa[j])
                    user_count[subject] = user_count.get(subject, 0) + int(sn[j])

                    gkey = str(gesture)
                    gesture_sum_abs[gkey] = gesture_sum_abs.get(gkey, 0.0) + float(sa[j])
                    gesture_count[gkey] = gesture_count.get(gkey, 0) + int(sn[j])

                    # Per-user per-group
                    if subject not in user_group_sum_abs:
                        user_group_sum_abs[subject] = {}
                        user_group_count[subject] = {}
                    for gname in group_indices:
                        user_group_sum_abs[subject][gname] = (
                            user_group_sum_abs[subject].get(gname, 0.0) + float(g_sa[gname][j])
                        )
                        user_group_count[subject][gname] = (
                            user_group_count[subject].get(gname, 0) + int(g_sn[gname][j])
                        )

            def _compute_stats(sum_abs: dict, count: dict) -> dict:
                per_group_maes = {}
                for key in sorted(sum_abs.keys()):
                    if count.get(key, 0) > 0:
                        mae = sum_abs[key] / count[key]
                        if np.isfinite(mae):
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

            user_stats = _compute_stats(user_sum_abs, user_count)
            gesture_stats = _compute_stats(gesture_sum_abs, gesture_count)

            stat: dict = {}
            if user_stats.get("count", 0) > 1:
                stat["per_user_mae_mean"] = user_stats["mean"]
                stat["per_user_mae_std"] = user_stats["std"]
                stat["per_user_count"] = user_stats["count"]
            if gesture_stats.get("count", 0) > 1:
                stat["per_gesture_mae_mean"] = gesture_stats["mean"]
                stat["per_gesture_mae_std"] = gesture_stats["std"]
                stat["per_gesture_count"] = gesture_stats["count"]

            # Per-user per-joint MAE stats
            for gname in group_indices:
                gmaes = []
                for user in sorted(user_group_sum_abs.keys()):
                    ug_sum = user_group_sum_abs[user].get(gname, 0.0)
                    ug_cnt = user_group_count[user].get(gname, 0)
                    if ug_cnt > 0:
                        ug_mae = ug_sum / ug_cnt
                        if np.isfinite(ug_mae):
                            gmaes.append(ug_mae)
                if gmaes:
                    stat[f"per_user_mae_{gname}_mean"] = float(np.mean(gmaes))
                    stat[f"per_user_mae_{gname}_std"] = float(np.std(gmaes))

            all_stats.append(stat)

        return all_stats

    def _evaluate_egoemg_pooled(self) -> dict[str, dict]:
        """Per-user & per-gesture stats pooled across left+right hands.

        Processes dataloader pairs (user/left+user/right, gesture/left+gesture/right,
        both/left+both/right), accumulating per-user and per-gesture sum_abs and
        count from both hands, then computes pooled per-user/per-gesture MAE stats.

        Returns a dict mapping pooled split name → stat dict.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = self.module
        module.to(device)
        module.eval()

        # Datasets are stored in order: user/left, user/right, gesture/left,
        # gesture/right, both/left, both/right
        datasets = self._egoemg_wrapped_datasets
        split_names = ["user", "gesture", "both"]
        pooled_stats: dict[str, dict] = {}

        # Cross-split accumulators for overall sample-level MAE.
        all_sum = 0.0
        all_cnt = 0

        for pair_idx, split_name in enumerate(split_names):
            left_ds = datasets[pair_idx * 2]
            right_ds = datasets[pair_idx * 2 + 1]

            # ── Build index metadata for both hands ──
            # We process left and right separately but pool by subject name.
            # Subjects are consistent across left/right datasets.
            user_sum_abs: dict[str, float] = {}
            user_count: dict[str, int] = {}
            gesture_sum_abs: dict[str, float] = {}
            gesture_count: dict[str, int] = {}
            overall_sum = 0.0
            overall_cnt = 0

            for hand, wrapper_ds in [("left", left_ds), ("right", right_ds)]:
                base_ds = wrapper_ds._base
                n = len(wrapper_ds)
                if n == 0:
                    continue

                index_meta: list[tuple[str, int]] = [
                    (base_ds.get_subject(i), base_ds.get_gesture_class(i))
                    for i in range(n)
                ]

                indexed_ds = _DatasetWithIndex(wrapper_ds)
                loader = DataLoader(
                    indexed_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=(device.type == "cuda"),
                    collate_fn=_egoemg_collate_fn,
                )

                for batch in loader:
                    indices = batch.pop("_idx", None)
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    with torch.no_grad():
                        preds, targets, mask = module.forward(batch)

                    C = preds.shape[1]
                    abs_per_time = (preds - targets).abs().sum(dim=1)
                    sample_abs = (abs_per_time * mask.float()).sum(dim=1)
                    sample_n = mask.float().sum(dim=1) * C
                    sa = sample_abs.cpu().numpy()
                    sn = sample_n.cpu().numpy()

                    overall_sum += float(sa.sum())
                    overall_cnt += int(sn.sum())
                    all_sum += float(sa.sum())
                    all_cnt += int(sn.sum())

                    batch_indices = indices.cpu().numpy() if indices is not None else range(len(sa))
                    for j in range(len(sa)):
                        if sn[j] == 0:
                            continue
                        idx = int(batch_indices[j])
                        subject, gesture = index_meta[idx]

                        user_sum_abs[subject] = user_sum_abs.get(subject, 0.0) + float(sa[j])
                        user_count[subject] = user_count.get(subject, 0) + int(sn[j])

                        gkey = str(gesture)
                        gesture_sum_abs[gkey] = gesture_sum_abs.get(gkey, 0.0) + float(sa[j])
                        gesture_count[gkey] = gesture_count.get(gkey, 0) + int(sn[j])

            # ── Compute stats ──
            def _compute_stats(sum_abs: dict, count: dict) -> dict:
                per_group_maes = {}
                for key in sorted(sum_abs.keys()):
                    if count.get(key, 0) > 0:
                        mae = sum_abs[key] / count[key]
                        if np.isfinite(mae):
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

            user_stats = _compute_stats(user_sum_abs, user_count)
            gesture_stats = _compute_stats(gesture_sum_abs, gesture_count)
            overall_mae = overall_sum / overall_cnt if overall_cnt > 0 else float("nan")

            stat: dict = {"test_mae": overall_mae}
            if user_stats.get("count", 0) > 1:
                stat["per_user_mae_mean"] = user_stats["mean"]
                stat["per_user_mae_std"] = user_stats["std"]
                stat["per_user_count"] = user_stats["count"]
            if gesture_stats.get("count", 0) > 1:
                stat["per_gesture_mae_mean"] = gesture_stats["mean"]
                stat["per_gesture_mae_std"] = gesture_stats["std"]
                stat["per_gesture_count"] = gesture_stats["count"]
            pooled_stats[split_name] = stat

        # ── Overall across all splits ──
        overall_mae = all_sum / all_cnt if all_cnt > 0 else float("nan")
        pooled_stats["overall"] = {"test_mae": overall_mae}

        return pooled_stats

    def _evaluate_emg2pose_single_pass(
        self, overall_results: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """One-pass inference: overall test_mae + per-user MAE + per-joint MAE.

        Replaces trainer.test() + per-user pass with a single model-forward
        pass. Uses num_workers > 0 for fast data loading (metadata flows
        through the default collate as tuple of strings).

        Returns (updated_overall_results, per_user_stats) as lists.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = self.module
        module.to(device)
        module.eval()

        # Pre-build per-group channel index arrays
        group_indices: dict[str, list[int]] = {}
        for finger in FINGERS:
            group_indices[finger] = [j.index for j in JOINTS if finger in j.groups]
        for pd_group in PD_GROUPS:
            group_indices[pd_group] = [j.index for j in JOINTS if pd_group in j.groups]

        per_user_stats: list[dict] = []

        for dl_idx, dataset in enumerate(
            tqdm(self._emg2pose_datasets, desc="Eval+Per-user")
        ):
            indexed_ds = _DatasetWithIndex(dataset)
            loader = DataLoader(
                indexed_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(device.type == "cuda"),
                prefetch_factor=2 if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
            )

            # Overall metric accumulators (batch-size-weighted)
            total_abs_sum = 0.0
            total_count = 0

            # Per-group accumulators
            group_sum_abs: dict[str, float] = {g: 0.0 for g in group_indices}
            group_count: dict[str, int] = {g: 0 for g in group_indices}

            # Per-user per-group accumulators
            user_group_sum_abs: dict[str, dict[str, float]] = {}
            user_group_count: dict[str, dict[str, int]] = {}

            # Per-user incremental accumulators
            user_sum_abs: dict[str, float] = {}
            user_count: dict[str, int] = {}

            for batch in loader:
                users_in_batch: tuple[str, ...] = batch.pop("_user", ())
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                batch.pop("_idx", None)
                n_samples = len(users_in_batch)

                with torch.no_grad():
                    preds, targets, mask = module.forward(batch)

                # ── Vectorized per-sample L1 errors (on GPU, one kernel) ──
                # preds/targets: (B, C, T), mask: (B, T)
                # L1Loss divides by (C × valid_time_steps); we match that.
                C = preds.shape[1]
                abs_per_time = (preds - targets).abs().sum(dim=1)  # (B, T)
                sample_abs = (abs_per_time * mask.float()).sum(dim=1)  # (B,)
                sample_n = mask.float().sum(dim=1) * C  # (B,) — element count
                sa = sample_abs.cpu().numpy()
                sn = sample_n.cpu().numpy()

                # Overall metrics
                total_abs_sum += float(sa.sum())
                total_count += int(sn.sum())

                # ── Per-group MAE (vectorized, one kernel per group) ──
                g_sa: dict[str, np.ndarray] = {}
                g_sn: dict[str, np.ndarray] = {}
                for gname, idxs in group_indices.items():
                    gC = len(idxs)
                    idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
                    g_abs_per_time = (preds[:, idxs_t] - targets[:, idxs_t]).abs().sum(dim=1)  # (B, T)
                    g_sample_abs = (g_abs_per_time * mask.float()).sum(dim=1)  # (B,)
                    g_sample_n = mask.float().sum(dim=1) * gC  # (B,)
                    group_sum_abs[gname] += float(g_sample_abs.sum().item())
                    group_count[gname] += int(g_sample_n.sum().item())
                    g_sa[gname] = g_sample_abs.cpu().numpy()
                    g_sn[gname] = g_sample_n.cpu().numpy()

                # Per-user accumulation (CPU-side, fast)
                for j in range(n_samples):
                    if sn[j] == 0:
                        continue
                    user = users_in_batch[j] if j < len(users_in_batch) else "unknown"
                    user_sum_abs[user] = user_sum_abs.get(user, 0.0) + float(sa[j])
                    user_count[user] = user_count.get(user, 0) + int(sn[j])

                    # Per-user per-group
                    if user not in user_group_sum_abs:
                        user_group_sum_abs[user] = {}
                        user_group_count[user] = {}
                    for gname in group_indices:
                        user_group_sum_abs[user][gname] = (
                            user_group_sum_abs[user].get(gname, 0.0) + float(g_sa[gname][j])
                        )
                        user_group_count[user][gname] = (
                            user_group_count[user].get(gname, 0) + int(g_sn[gname][j])
                        )

            # ── Overall MAE for this dataloader ──
            overall_mae = total_abs_sum / total_count if total_count > 0 else float("nan")
            if dl_idx < len(overall_results) and np.isfinite(overall_mae):
                overall_results[dl_idx]["test_mae"] = overall_mae

            # ── Per-group MAE ──
            for gname in group_indices:
                gmae = group_sum_abs[gname] / group_count[gname] if group_count[gname] > 0 else float("nan")
                key = f"test_mae_{gname}"
                if dl_idx < len(overall_results) and np.isfinite(gmae):
                    overall_results[dl_idx][key] = gmae

            # ── Per-user MAE stats (overall) ──
            per_user_maes: dict[str, float] = {}
            for user in sorted(user_sum_abs.keys()):
                if user_count.get(user, 0) > 0:
                    mae = user_sum_abs[user] / user_count[user]
                    if np.isfinite(mae):
                        per_user_maes[user] = mae

            if not per_user_maes:
                per_user_stats.append({})
                continue
            values = list(per_user_maes.values())
            stats: dict = {
                "per_user_count": len(values),
                "per_user_mae_mean": float(np.mean(values)),
                "per_user_mae_std": float(np.std(values)),
            }

            # ── Per-user per-group MAE stats ──
            for gname in group_indices:
                gmaes = []
                for user in sorted(user_group_sum_abs.keys()):
                    ug_sum = user_group_sum_abs[user].get(gname, 0.0)
                    ug_cnt = user_group_count[user].get(gname, 0)
                    if ug_cnt > 0:
                        ug_mae = ug_sum / ug_cnt
                        if np.isfinite(ug_mae):
                            gmaes.append(ug_mae)
                if gmaes:
                    stats[f"per_user_mae_{gname}_mean"] = float(np.mean(gmaes))
                    stats[f"per_user_mae_{gname}_std"] = float(np.std(gmaes))

            per_user_stats.append(stats)

        return overall_results, per_user_stats

    def create_results_df(self, results: list[dict[str, float]]) -> pd.DataFrame:
        """Convert results list to a dataframe with all condition information."""
        if self.dataset_type == "egoemg":
            records = []
            for name, result in zip(self._egoemg_group_names, results):
                record = {"split": name}
                result = {k.split("/")[0]: v for k, v in result.items()}
                record.update(result)
                records.append(record)
            return pd.DataFrame(records)

        records = []
        for (vals, _), result in zip(self.groupby, results):
            record = dict(zip(self.conditions, vals))
            result = {k.split("/")[0]: v for k, v in result.items()}
            record.update(result)
            records.append(record)
        return pd.DataFrame(records)

    def evaluate(self) -> pd.DataFrame:
        """Run analysis for split."""
        # ── emg2pose path with per_user: single-pass for speed ─────────
        # Avoids a second model-forward pass by computing test_mae and
        # per-user stats together.
        if self.per_user and self.dataset_type == "emg2pose":
            blanks = [{} for _ in self.dataloaders]
            init_df = self.create_results_df(blanks)
            # Run the single-pass evaluation
            updated_rows, per_user_stats = self._evaluate_emg2pose_single_pass(
                [row.to_dict() for _, row in init_df.iterrows()]
            )
            # Rebuild DataFrame from updated rows, then add per-user columns
            results_df = pd.DataFrame(updated_rows)
            for i, stat in enumerate(per_user_stats):
                if i < len(results_df):
                    for k, v in stat.items():
                        results_df.at[i, k] = v
            self._print_per_user_summary(results_df)
            return results_df

        trainer_cfg = dict(self.config.trainer)
        trainer_cfg["logger"] = False  # avoid hparams serialization issues
        trainer_cfg["devices"] = 1  # force single GPU for evaluation
        trainer = pl.Trainer(**trainer_cfg)
        results = trainer.test(self.module, dataloaders=self.dataloaders, verbose=True)
        results_df = self.create_results_df(results)

        # ── Per-group stats for EgoEMG path ────────────────────────────
        if self.per_group_stats and self.dataset_type == "egoemg":
            per_group_stats = self._evaluate_egoemg_per_group()
            for i, stat in enumerate(per_group_stats):
                if i < len(results_df):
                    for k, v in stat.items():
                        results_df.at[i, k] = v
            self._print_per_group_summary(results_df)

            # ── Pooled left+right stats ──
            pooled = self._evaluate_egoemg_pooled()
            print("\n" + "=" * 60)
            print("Pooled left+right per-user mean ± std")
            print("=" * 60)
            for split_name in ["gesture", "user", "both", "overall"]:
                s = pooled.get(split_name, {})
                if split_name == "overall":
                    print(f"  {'overall':12s}  test_mae={s['test_mae']:.4f}")
                elif s.get("per_user_count", 0) > 0:
                    print(f"  {split_name:12s}  test_mae={s['test_mae']:.4f}  "
                          f"per-user  mae={s['per_user_mae_mean']:.4f} ± "
                          f"{s['per_user_mae_std']:.4f}  (n={int(s['per_user_count'])})")
            print("=" * 60 + "\n")

        return results_df

    def _print_per_group_summary(self, df: pd.DataFrame):
        """Print per-user and per-gesture mean ± std for each split."""
        print("\n" + "=" * 60)
        print("Per-group mean angular error ± std across users/gestures")
        print("=" * 60)
        for _, row in df.iterrows():
            label = row.get("split", "?")
            user_str = ""
            if row.get("per_user_count", 0) > 0:
                user_str = (f"per-user  mae={row['per_user_mae_mean']:.6f} ± "
                            f"{row['per_user_mae_std']:.6f} (n={int(row['per_user_count'])})")
            gesture_str = ""
            if row.get("per_gesture_count", 0) > 0:
                gesture_str = (f"per-gesture  mae={row['per_gesture_mae_mean']:.6f} ± "
                               f"{row['per_gesture_mae_std']:.6f} (n={int(row['per_gesture_count'])})")
            if user_str or gesture_str:
                print(f"  {label:20s}  {user_str}  {gesture_str}")
        print("=" * 60 + "\n")

    def _print_per_user_summary(self, df: pd.DataFrame):
        """Print per-user mean ± std for each generalization condition."""
        print("\n" + "=" * 60)
        print("Per-user mean angular error ± std across users")
        print("=" * 60)
        for _, row in df.iterrows():
            label = row.get("generalization", row.get("split", "?"))
            if row.get("per_user_count", 0) > 0:
                print(f"  {label:20s}  mae={row['per_user_mae_mean']:.6f} ± "
                      f"{row['per_user_mae_std']:.6f} (n={int(row['per_user_count'])})")
        print("=" * 60 + "\n")


def _egoemg_collate_fn(batch):
    """Custom collate for EgoEMG datasets to match regression module input format.

    Supports both:
    - Raw EgoEmgMemmapDataset: joint_angles + label_valid_mask (1D)
    - PretrainWrapperDataset: angle_target + angle_mask (2D: C, T)
    """
    import torch
    import numpy as np
    from torch.utils.data._utils.collate import default_collate

    if not batch:
        return {}

    processed = []
    for sample in batch:
        emg = sample["emg"]  # (C, T)
        if isinstance(emg, np.ndarray):
            emg = torch.as_tensor(emg, dtype=torch.float32)

        time_len = emg.shape[-1]

        # Support both raw and PretrainWrapperDataset outputs
        if "angle_target" in sample:
            # PretrainWrapperDataset path
            ja = torch.as_tensor(sample["angle_target"], dtype=torch.float32)
            # angle_mask is (C, T) - reduce to per-time mask (any channel valid = time valid)
            mask_2d = torch.as_tensor(sample["angle_mask"], dtype=torch.bool)
            mask = mask_2d.any(dim=0)  # (T,)
        else:
            # Raw EgoEmgMemmapDataset path
            ja = sample.get("joint_angles")
            if ja is not None:
                if isinstance(ja, np.ndarray):
                    ja = torch.as_tensor(ja, dtype=torch.float32)
            else:
                ja = torch.zeros(time_len, dtype=torch.float32)
            mask = torch.ones(time_len, dtype=torch.bool)

        item = {
            "emg": emg,
            "joint_angles": ja,
            "label_valid_mask": mask,
        }
        if "_idx" in sample:
            item["_idx"] = sample["_idx"]
        processed.append(item)

    return default_collate(processed)


def _run_center_frame_eval(config: DictConfig) -> pd.DataFrame:
    """Evaluate a vision/fusion checkpoint on unified center frames.

    Uses the in-process center-frame evaluator (``egoemg.center_frame``) with
    the already-composed Hydra config, so it works inside the enclosing
    ``@hydra.main`` without re-entering Hydra.
    """
    from egoemg import center_frame

    ckpt_path = config.get("center_frame_ckpt") or config.checkpoint
    if not ckpt_path:
        raise ValueError("center_frame_eval requires a checkpoint (center_frame_ckpt or checkpoint)")
    memmap_dir = (
        config.get("egoemg_unified_memmap_dir")
        or config.get("egoemg_memmap_dir")
        or "data/EgoEMG_memmap"
    )
    center_window = config.get("center_frame_window_length")
    results = center_frame.eval_center_frame(
        config, str(ckpt_path), Path(memmap_dir), center_window
    )

    rows = []
    for hand, r in results.items():
        rows.append(
            {
                "hand": hand,
                "test_mae": r.get("overall_mae"),
                "n_valid": r.get("n_valid"),
            }
        )
    df = pd.DataFrame(rows)
    results_filename = _results_output_path()
    print(f"Saving center-frame results to {results_filename}")
    df.to_csv(results_filename, index=False)
    return df


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def cli(config: DictConfig):
    if config.get("center_frame_eval", False):
        return _run_center_frame_eval(config)

    dataset_type = config.get("dataset_type") or infer_dataset_type(config)
    per_user = config.get("per_user", False)
    per_group_stats = config.get("per_group_stats", False)
    evaluation = EMG2PoseEvaluation(
        config=config,
        checkpoint=config.checkpoint,
        conditions=["generalization"],
        window_length=config.get("datamodule", {}).get("val_test_window_length", 10_000) if hasattr(config, 'datamodule') else 10_000,
        split="test",
        skip_ik_failures=True,
        dataset_type=dataset_type,
        per_user=per_user,
        per_group_stats=per_group_stats,
    )
    results_df = evaluation.evaluate()

    results_filename = _results_output_path()
    print(f"Saving results to {results_filename}")
    results_df.to_csv(results_filename, index=False)


if __name__ == "__main__":
    cli()
