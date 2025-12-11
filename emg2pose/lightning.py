# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from emg2pose import utils
from emg2pose.data import WindowedEmgDataset
from emg2pose.metrics import get_default_metrics
from emg2pose.pose_modules import BasePoseModule
from hydra.utils import instantiate

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        val_test_window_length: int | None = None,
        skip_ik_failures: bool = False,
        allow_mask_recompute: bool = False,
        treat_interpolated_as_valid: bool = True,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.skip_ik_failures = skip_ik_failures
        self.allow_mask_recompute = allow_mask_recompute
        self.treat_interpolated_as_valid = treat_interpolated_as_valid

    def setup(self, stage: str | None = None) -> None:
        # train
        self.train_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.train_transforms,
                window_length=self.window_length,
                padding=self.padding,
                jitter=True,
                skip_ik_failures=self.skip_ik_failures,
                allow_mask_recompute=self.allow_mask_recompute,
                treat_interpolated_as_valid=self.treat_interpolated_as_valid,
            )
            for hdf5_path in tqdm(self.train_sessions, desc="Building train datasets")
        ])

        # val
        self.val_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.val_transforms,
                window_length=self.val_test_window_length,
                padding=self.padding,
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
                allow_mask_recompute=self.allow_mask_recompute,
                treat_interpolated_as_valid=self.treat_interpolated_as_valid,
            )
            for hdf5_path in tqdm(self.val_sessions, desc="Building val datasets")
        ])

        # test
        self.test_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.test_transforms,
                window_length=self.val_test_window_length,
                padding=(0, 0),
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
                allow_mask_recompute=self.allow_mask_recompute,
                treat_interpolated_as_valid=self.treat_interpolated_as_valid,
            )
            for hdf5_path in tqdm(self.test_sessions, desc="Building test datasets")
        ])


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )


class SlidingEmgDataModule(pl.LightningDataModule):
    """Slide windows sequentially without excluding IK failures.

    Loss functions should use the provided `no_ik_failure` (and optionally
    `interpolated_mask`) to mask invalid labels instead of dropping windows.
    """

    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        val_test_window_length: int | None = None,
        allow_mask_recompute: bool = False,
        treat_interpolated_as_valid: bool = True,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.allow_mask_recompute = allow_mask_recompute
        self.treat_interpolated_as_valid = treat_interpolated_as_valid

    def setup(self, stage: str | None = None) -> None:
        # train
        self.train_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.train_transforms,
                window_length=self.window_length,
                padding=self.padding,
                jitter=True,
                skip_ik_failures=False,
                allow_mask_recompute=self.allow_mask_recompute,
                treat_interpolated_as_valid=self.treat_interpolated_as_valid,
            )
            for hdf5_path in tqdm(self.train_sessions, desc="Building train datasets")
        ])

        # val
        self.val_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.val_transforms,
                window_length=self.val_test_window_length,
                padding=self.padding,
                jitter=False,
                skip_ik_failures=False,
                allow_mask_recompute=self.allow_mask_recompute,
                treat_interpolated_as_valid=self.treat_interpolated_as_valid,
            )
            for hdf5_path in tqdm(self.val_sessions, desc="Building val datasets")
        ])

        # test
        self.test_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.test_transforms,
                window_length=self.val_test_window_length,
                padding=(0, 0),
                jitter=False,
                skip_ik_failures=False,
                allow_mask_recompute=self.allow_mask_recompute,
                treat_interpolated_as_valid=self.treat_interpolated_as_valid,
            )
            for hdf5_path in tqdm(self.test_sessions, desc="Building test datasets")
        ])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )


class CachedWindowDataset(Dataset):
    def __init__(self, cache_dir: Path, transform=None) -> None:
        super().__init__()
        manifest_path = cache_dir.joinpath("manifest.csv")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest file at {manifest_path}")

        self.manifest = pd.read_csv(manifest_path)
        if self.manifest.empty:
            raise ValueError(f"Manifest {manifest_path} is empty.")

        self.cache_dir = cache_dir
        self.transform = transform

        self.effective_length = int(self.manifest["effective_length"].iloc[0])
        self.window_length = int(self.manifest["window_length"].iloc[0])
        self.emg_channels = int(self.manifest["emg_channels"].iloc[0])
        self.joint_dims = int(self.manifest["joint_dims"].iloc[0])
        self._window_start_idx = self.manifest["window_start_idx"].to_numpy(np.int64)
        self._window_end_idx = self.manifest["window_end_idx"].to_numpy(np.int64)

        num_windows = len(self.manifest)
        emg_path = cache_dir.joinpath("emg.f32")
        angles_path = cache_dir.joinpath("joint_angles.f32")
        mask_path = cache_dir.joinpath("no_ik_mask.u1")
        if not emg_path.exists() or not angles_path.exists() or not mask_path.exists():
            raise FileNotFoundError(
                f"Expected cache files emg.f32, joint_angles.f32, and no_ik_mask.u1 in {cache_dir}"
            )
        self.emg = np.memmap(
            emg_path,
            mode="r+",
            dtype=np.float32,
            shape=(num_windows, self.effective_length, self.emg_channels),
        )
        self.joint_angles = np.memmap(
            angles_path,
            mode="r+",
            dtype=np.float32,
            shape=(num_windows, self.effective_length, self.joint_dims),
        )
        self.mask = np.memmap(
            mask_path,
            mode="r+",
            dtype=np.uint8,
            shape=(num_windows, self.effective_length),
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        emg_tensor = torch.from_numpy(self.emg[idx])
        emg_input = {"emg": emg_tensor}
        if self.transform is not None:
            try:
                emg_tensor = self.transform(emg_input)
            except Exception as exc:
                raise RuntimeError(
                    "CachedWindowDataset transform failed. "
                    "Ensure transforms expect a torch.Tensor EMG window."
                ) from exc
        emg_window = emg_tensor.transpose(0, 1).contiguous()

        joint_angles = torch.from_numpy(self.joint_angles[idx]).transpose(0, 1).contiguous()
        mask = torch.from_numpy(self.mask[idx].astype(np.bool_, copy=False))

        return {
            "emg": emg_window,
            "joint_angles": joint_angles,
            "no_ik_failure": mask,
            "window_start_idx": int(self._window_start_idx[idx]),
            "window_end_idx": int(self._window_end_idx[idx]),
        }


class CachedWindowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cache_root: str | Path,
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path] | None = None,
        val_sessions: Sequence[Path] | None = None,
        test_sessions: Sequence[Path] | None = None,
        skip_ik_failures: bool = False,
        allow_missing_splits: bool = False,
        window_length: int | None = None,
        val_test_window_length: int | None = None,
    ) -> None:
        super().__init__()
        self.cache_root = Path(cache_root).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_missing_splits = allow_missing_splits

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
        self._datasets: dict[str, Dataset | None] = {
            "train": None,
            "val": None,
            "test": None,
        }

    def _build_dataset(self, split: str, transform):
        split_dir = self.cache_root.joinpath(split)
        if not split_dir.exists():
            if self.allow_missing_splits:
                return None
            raise FileNotFoundError(
                f"Unable to find cached split '{split}' at {split_dir}. "
                "Run scripts/cache_windows.py first or enable allow_missing_splits."
            )
        return CachedWindowDataset(split_dir, transform=transform)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._datasets["train"] = self._build_dataset("train", self.train_transforms)
            self._datasets["val"] = self._build_dataset("val", self.val_transforms)
        if stage in (None, "test"):
            self._datasets["test"] = self._build_dataset("test", self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        dataset = self._datasets.get("train")
        if dataset is None:
            raise RuntimeError("Training dataset is not available. Check cached data.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=2
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._datasets.get("val")
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self._datasets.get("test")
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )


class Emg2PoseModule(pl.LightningModule):
    def __init__(
        self,
        network_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        provide_initial_pos: bool = False,
        loss_weights: dict[str, float] | None = None,
        use_interpolated_as_valid: bool = True,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.model: BasePoseModule = instantiate(network_conf, _convert_="all")
        self.provide_initial_pos = provide_initial_pos
        self.loss_weights = loss_weights or {"mae": 1}
        self.use_interpolated_as_valid = use_interpolated_as_valid

        # TODO: add metrics to Hydra config instead
        self.metrics_list = get_default_metrics()

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward(batch, self.provide_initial_pos)

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str = "train"
    ) -> torch.Tensor:

        # Generate predictions
        batch["no_ik_failure"] = self.update_ik_failure_mask(batch["no_ik_failure"])
        preds, targets, no_ik_failure = self.forward(batch)

        # Align interpolated mask if provided
        aligned_interp = None
        if "interpolated_mask" in batch:
            interp = batch["interpolated_mask"]
            start = self.model.left_context
            stop = None if self.model.right_context == 0 else -self.model.right_context
            interp = interp[..., slice(start, stop)]
            aligned_interp = self.model.align_mask(interp, targets.shape[-1])

        # Build final loss/metric mask
        valid_mask = self.build_valid_mask(
            base_mask=no_ik_failure,
            targets=targets,
            interpolated_mask=aligned_interp,
        )

        # Compute metrics
        metrics = {}
        for metric in self.metrics_list:
            metrics.update(metric(preds, targets, valid_mask, stage))
        self.log_dict(metrics, sync_dist=True)

        # Compute loss
        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            loss += metrics[f"{stage}_{loss_name}"] * weight
        self.log(f"{stage}_loss", loss, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(
        self, batch, batch_idx, dataloader_idx: int | None = None
    ) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )

    def update_ik_failure_mask(self, no_ik_failure: torch.Tensor) -> torch.Tensor:
        """Update the mask to only include samples where there are no ik failures."""

        # Mask out samples where the initial position is zero, because state
        # initialization doesn't work under these conditions. Note that the initial
        # position is the left_context'th sample, not the 0th sample.
        mask = no_ik_failure.clone()

        if self.provide_initial_pos:
            mask[~mask[:, self.model.left_context]] = False

        if mask.sum() == 0:
            log.warning("All samples masked out due to missing initial state!")

        return mask

    def build_valid_mask(
        self,
        base_mask: torch.Tensor,
        targets: torch.Tensor,
        interpolated_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine base IK mask, optional interpolated mask, and finite-value check."""
        mask = base_mask.bool()

        if interpolated_mask is not None and self.use_interpolated_as_valid:
            mask = mask | interpolated_mask.bool()

        # Drop any timestep containing NaN/Inf in targets across joints
        finite = torch.isfinite(targets).all(dim=1)
        mask = mask & finite

        # Warn if everything is masked to avoid empty tensors in losses
        if mask.sum() == 0:
            log.warning("All samples masked out after combining IK/interp/finite checks.")

        return mask
