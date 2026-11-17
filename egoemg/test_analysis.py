# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Offline evaluation for EMG2Pose corpus and EgoEMG session datasets."""

import json
import sys
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
from egoemg.datasets.corpora.emg2pose import Emg2PoseDataset
from egoemg.training.prediction import EmgPredictionModule
from egoemg.train import _extract_state_dict, _is_pretrain_checkpoint, make_lightning_module
from egoemg.augmentation.transforms import Compose


def _results_output_path() -> str:
    """Where to write the results.csv eval artifact.

    Uses the Hydra run output dir when available, so eval runs never clobber
    a results.csv in the working directory (repo root). Standalone evaluation
    falls back to logs/evaluation/standalone.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        output_dir = os.path.join(os.getcwd(), "logs", "evaluation", "standalone")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.abspath(os.path.join(output_dir, "results.csv"))


def _session_results_output_path() -> str:
    return os.path.join(os.path.dirname(_results_output_path()), "session_results.csv")


def _session_summary_output_path() -> str:
    return os.path.join(os.path.dirname(_results_output_path()), "session_summary.csv")


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
            if target.rsplit(".", 1)[-1].startswith("EgoEmgSession"):
                return "egoemg_session_v2"

    return "emg2pose"


@dataclass
class _SessionSums:
    windows: int = 0
    finger_error: float = 0.0
    finger_count: int = 0
    wrist_error: float = 0.0
    wrist_count: int = 0

    def add(self, other: "_SessionSums") -> None:
        self.windows += other.windows
        self.finger_error += other.finger_error
        self.finger_count += other.finger_count
        self.wrist_error += other.wrist_error
        self.wrist_count += other.wrist_count

    def row(self) -> dict[str, float | int]:
        total_count = self.finger_count + self.wrist_count
        total_error = self.finger_error + self.wrist_error
        return {
            "windows": self.windows,
            "valid_values": total_count,
            "valid_finger_values": self.finger_count,
            "valid_wrist_values": self.wrist_count,
            "test_mae": total_error / total_count if total_count else float("nan"),
            "finger_mae": (
                self.finger_error / self.finger_count
                if self.finger_count else float("nan")
            ),
            "wrist_mae": (
                self.wrist_error / self.wrist_count
                if self.wrist_count else float("nan")
            ),
        }


@dataclass
class EMG2PoseEvaluation:
    """
    Run offline evaluation using EMG2Pose corpus or EgoEMG session datasets.

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
    dataset_type: str = "emg2pose"  # emg2pose or egoemg_session_v2
    per_user: bool = False

    def __post_init__(self):
        self.data_dir = self.config.data_location
        self.batch_size = self.config.get("batch_size", self.batch_size)
        self.num_workers = self.config.get("num_workers", 12)
        inferred_dataset_type = infer_dataset_type(self.config)
        if self.dataset_type == "emg2pose":
            self.dataset_type = inferred_dataset_type
        if self.dataset_type not in {"emg2pose", "egoemg_session_v2"}:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        # Determine memmap directory
        self.memmap_dir = self._find_memmap_dir()

        self.is_pretrain = False  # set by get_module()
        self.module = self.get_module()

        if self.dataset_type == "egoemg_session_v2":
            self.dataloaders = self._get_session_v2_dataloaders()
        else:
            self.df = self.get_corpus_df()
            self.groupby = self.df.groupby(self.conditions)
            self.dataloaders = self.get_dataloaders()

    def _find_memmap_dir(self) -> str:
        """Find the EMG2Pose corpus memmap from data_location."""
        data_dir = self.data_dir.rstrip("/")
        # Check if data_location is already memmap
        if os.path.exists(os.path.join(data_dir, "manifest.json")):
            return data_dir
        if self.dataset_type == "egoemg_session_v2":
            return data_dir
        raise FileNotFoundError(f"Expected an explicit memmap directory: {data_dir}")

    def get_corpus_df(self) -> pd.DataFrame:
        """Load metadata and filter by split."""
        # Try loading from metadata.npz (memmap format)
        possible_npz_paths = [
            os.path.join(self.data_dir, "metadata.npz"),
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
        # NPZ indexing decompresses an array on every access. Materialize the
        # small metadata columns once, not once per recording in the loop.
        with np.load(npz_path) as archive:
            data = {key: archive[key] for key in archive.files
                    if key.startswith(('session_', 'users_', 'stages_', 'splits_'))}

        def decode(arr):
            return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]

        records = []
        n_sessions = len(data["session_start_idx"])

        users = decode(data["users_user"])
        stages = decode(data.get("stages_stage", []))
        splits = decode(data.get("splits_split", ["train", "val", "test"]))

        for i in range(n_sessions):
            record = {
                "session_index": i,
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
        # Checkpoints record OmegaConf hyperparameters that PyTorch 2.6+'s
        # weights-only default cannot deserialize (trusted local artifact).
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
            # Evaluation machines may be offline; trained weights come from
            # the checkpoint itself, so suppress ImageNet/DINO init downloads.
            os.environ.setdefault("EGOEMG_NO_PRETRAINED_DOWNLOAD", "1")
            module = module.__class__.load_from_checkpoint(ckpt_path, **kwargs)

        module.eval()
        return module


    def _get_session_v2_dataloaders(self) -> list[DataLoader]:
        """Use the experiment's published test datasets and window settings."""
        dataset_cfg = self.config.dataset.test
        splits = self.config.get("egoemg_test_splits", None)
        datasets = [
            instantiate(cfg, transform=None,
                        **({"allowed_splits": list(splits)} if splits is not None else {}))
            for cfg in dataset_cfg
        ]
        self._session_v2_datasets = datasets
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=2 if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
            )
            for dataset in datasets
        ]

    def get_dataloaders(self) -> list[DataLoader]:
        """Get list of dataloaders for emg2pose dataset."""
        if hasattr(self.config, 'transforms') and self.config.transforms is not None:
            if self.split in self.config.transforms and self.config.transforms[self.split] is not None:
                transforms = Compose(instantiate(self.config.transforms[self.split], _convert_="all"))
            else:
                from egoemg.augmentation.transforms import ExtractStructuredFields
                transforms = ExtractStructuredFields()
        else:
            from egoemg.augmentation.transforms import ExtractStructuredFields
            transforms = ExtractStructuredFields()

        print("Creating dataloaders for each condition.")
        dataloaders = []
        self._emg2pose_datasets = []  # stored for per-user evaluation
        for _, df_ in tqdm(self.groupby):
            ds_kwargs = dict(
                memmap_dir=self.memmap_dir,
                # Match the training/test datamodule windowing exactly; model
                # context is handled internally during forward/alignment.
                window_length=self.window_length,
                stride=self.window_length,
                padding=(0, 0),
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
                allowed_splits=[self.split],
                transform=transforms,
            )
            # session_id identifies a recording session, not an individual
            # recording-side entry. Selecting it can pull training recordings
            # and other generalization groups into a held-out evaluation.
            if "session_index" in df_.columns:
                ds_kwargs["allowed_session_indices"] = df_["session_index"].tolist()
            elif "filename" in df_.columns:
                ds_kwargs["allowed_sessions"] = df_["filename"].tolist()
            else:
                raise ValueError("EMG2Pose evaluation requires exact record identifiers")

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
        records = []
        for (vals, _), result in zip(self.groupby, results):
            record = dict(zip(self.conditions, vals))
            result = {k.replace("/", "_"): v for k, v in result.items()}
            record.update(result)
            records.append(record)
        return pd.DataFrame(records)

    def _evaluate_session_v2(self) -> pd.DataFrame:
        """One inference pass for split/hand totals and per-session metrics."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = self.module.to(device).eval()
        session_sums: dict[tuple[str, str, str], _SessionSums] = {}
        metadata: dict[str, tuple[str, str]] = {}
        for dataset in self._session_v2_datasets:
            for session in getattr(dataset, "_sessions", []):
                metadata[session.session_id] = (session.source_group, session.subject)
        dataset_root = getattr(self._session_v2_datasets[0], "dataset_dir", None)
        entries_by_id = {}
        if dataset_root is not None:
            manifest = json.loads(
                (Path(dataset_root) / "dataset.json").read_text(encoding="utf-8")
            )
            for entry in manifest["sessions"]:
                session = json.loads(
                    (Path(dataset_root) / entry["path"]).read_text(encoding="utf-8")
                )
                metadata[session["session_id"]] = (
                    entry["source_group"], str(session["subject"])
                )
                entries_by_id[session["session_id"]] = entry

        with torch.inference_mode():
            for dataset, loader in zip(self._session_v2_datasets, self.dataloaders):
                for batch in tqdm(loader, desc=f"Session-v2 {getattr(dataset, 'target_hand', 'vision')}"):
                    session_ids = batch.pop("episode_id")
                    split_ids = batch.pop("split_id", None)
                    held_user = batch.pop("held_out_user", None)
                    held_gesture = batch.pop("held_out_stage", None)
                    bimanual = getattr(getattr(module, "model", module),
                                       "bimanual_output", False)
                    if bimanual:
                        hand_indices = [0, 1] * len(session_ids)
                        session_ids = [s for s in session_ids for _ in range(2)]
                        split_ids = split_ids.reshape(-1)
                    else:
                        hand_indices = (batch["target_hand_index"].tolist()
                                        if "target_hand_index" in batch else
                                        [0 if dataset.target_hand == "left" else 1] * len(session_ids))
                    batch.pop("target_hand", None)
                    batch.pop("dataset_name", None)
                    batch = {
                        key: value.to(device, non_blocking=True)
                        if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    preds, targets, mask = module.forward(batch)
                    if mask.ndim == 2:
                        mask = mask[:, None, :].expand_as(preds)
                    if mask.shape != preds.shape:
                        raise ValueError(
                            f"Prediction/mask shape mismatch: {preds.shape}, {mask.shape}"
                        )
                    mask = mask.bool().clone()
                    for row, session_id in enumerate(session_ids):
                        # Match EmgPredictionModule._step: historical ShowEE
                        # has no trusted wrist supervision in this metric.
                        if metadata[session_id][0] == "showee" and mask.shape[1] > 20:
                            mask[row, 20:, :] = False
                    error = torch.where(mask, (preds - targets).abs(), 0.0)
                    finger_error = error[:, :20].sum((1, 2)).cpu().tolist()
                    finger_count = mask[:, :20].sum((1, 2)).cpu().tolist()
                    wrist_error = error[:, 20:].sum((1, 2)).cpu().tolist()
                    wrist_count = mask[:, 20:].sum((1, 2)).cpu().tolist()
                    for row, session_id in enumerate(session_ids):
                        from egoemg.datasets.session.schema import (SPLIT_IDS)
                        if bimanual and not (finger_count[row] + wrist_count[row]):
                            continue
                        split_id = (int(split_ids[row]) if split_ids is not None else
                                    int(held_user[row]) + 2 * int(held_gesture[row]))
                        split = {value: key for key, value in SPLIT_IDS.items()}[split_id]
                        key = (split, ("left", "right")[hand_indices[row]], session_id)
                        sums = session_sums.setdefault(key, _SessionSums())
                        sums.windows += 1
                        sums.finger_error += float(finger_error[row])
                        sums.finger_count += int(finger_count[row])
                        sums.wrist_error += float(wrist_error[row])
                        sums.wrist_count += int(wrist_count[row])

        grouped: dict[tuple[str, str], _SessionSums] = {}
        pooled: dict[tuple[str, str], _SessionSums] = {}
        by_session: dict[str, _SessionSums] = {}
        session_splits: dict[str, set[str]] = {}
        rows = []
        for (split, hand, session_id), sums in sorted(session_sums.items()):
            source_group, subject = metadata[session_id]
            rows.append({"split": split, "session_id": session_id,
                         "subject": subject, "source_group": source_group,
                         "hand": hand, **sums.row()})
            grouped.setdefault((split, hand), _SessionSums()).add(sums)
            pooled.setdefault((split, session_id), _SessionSums()).add(sums)
            by_session.setdefault(session_id, _SessionSums()).add(sums)
            session_splits.setdefault(session_id, set()).add(split)
        for (split, session_id), sums in sorted(pooled.items()):
            source_group, subject = metadata[session_id]
            rows.append({"split": split, "session_id": session_id,
                         "subject": subject, "source_group": source_group,
                         "hand": "pooled", **sums.row()})
        session_df = pd.DataFrame(rows)
        session_path = _session_results_output_path()
        session_df.to_csv(session_path, index=False)
        print(f"Saved {len(session_df)} per-session rows to {session_path}")
        summary_rows = []
        for session_id in sorted(metadata):
            sums = by_session.get(session_id, _SessionSums())
            source_group, subject = metadata[session_id]
            status = "evaluated" if sums.windows else "no_evaluable_windows"
            if not sums.windows and session_id in entries_by_id:
                entry = entries_by_id[session_id]
                folder = (Path(dataset_root) / entry["path"]).parent
                session = json.loads(
                    (folder / "session.json").read_text(encoding="utf-8")
                )
                split_arrays = [
                    np.memmap(
                        folder / hand["arrays"]["split_id"]["path"],
                        dtype=hand["arrays"]["split_id"]["dtype"],
                        mode="r",
                        shape=tuple(hand["arrays"]["split_id"]["shape"]),
                    )
                    for hand in session["hands"].values() if hand
                ]
                if split_arrays and all(np.all(x == -1) for x in split_arrays):
                    status = "unassigned_split"
            summary_rows.append({
                "session_id": session_id,
                "subject": subject,
                "source_group": source_group,
                "splits": ",".join(sorted(session_splits.get(session_id, set()))),
                "status": status,
                **sums.row(),
            })
        summary_path = _session_summary_output_path()
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Saved {len(summary_rows)} session summaries to {summary_path}")
        return pd.DataFrame([
            {"split": split, "hand": hand, **sums.row()}
            for (split, hand), sums in sorted(grouped.items())
        ])

    def evaluate(self) -> pd.DataFrame:
        """Run analysis for split."""
        if self.dataset_type == "egoemg_session_v2":
            return self._evaluate_session_v2()
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

        return results_df


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


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def cli(config: DictConfig):
    if config.get("center_frame_eval", False):
        raise ValueError("Episode center-frame evaluation was removed; use session evaluation")

    dataset_type = config.get("dataset_type") or infer_dataset_type(config)
    per_user = config.get("per_user", False)
    evaluation = EMG2PoseEvaluation(
        config=config,
        checkpoint=config.checkpoint,
        conditions=["generalization"],
        window_length=config.get("datamodule", {}).get("val_test_window_length", 10_000) if hasattr(config, 'datamodule') else 10_000,
        split="test",
        skip_ik_failures=True,
        dataset_type=dataset_type,
        per_user=per_user,
    )
    results_df = evaluation.evaluate()

    results_filename = _results_output_path()
    print(f"Saving results to {results_filename}")
    results_df.to_csv(results_filename, index=False)


if __name__ == "__main__":
    if any(x.split("=", 1)[0] == "--checkpoint-dir" for x in sys.argv[1:]):
        from egoemg.evaluate import main

        main()
    else:
        cli()
