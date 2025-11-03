#!/usr/bin/env python

# Copyright (c) Meta Platforms
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import hydra
import pandas as pd
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from torch.utils.data import ConcatDataset, DataLoader, Subset

from emg2pose.lightning import CachedWindowDataset
from emg2pose.train import make_lightning_module
from emg2pose.transforms import Compose


@dataclass
class CachedEMG2PoseEvaluation:
    """
    Offline evaluation over cached datasets.
    Metrics are aggregated per group (e.g. per generalization / user).
    """

    config: DictConfig
    checkpoint: str
    cache_root: Path
    conditions: list[str]
    split: str = "test"
    batch_size: int = 32

    def __post_init__(self) -> None:
        self.df = self._load_split_dataframe()
        self.groupby = self.df.groupby(self.conditions)
        self.module = self._load_module()

    def _load_split_dataframe(self) -> pd.DataFrame:
        manifest_path = self.cache_root.joinpath(self.split, "manifest.csv")
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cached manifest not found for split '{self.split}': {manifest_path}"
            )

        manifest = pd.read_csv(manifest_path)
        metadata_cols = [
            "session",
            "file_name",
            "user",
            "stage",
            "generalization",
            "split",
            "global_index",
        ]
        missing = [col for col in ["generalization", "split"] if col not in manifest]
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: {missing}"
            )
        metadata = manifest[metadata_cols].copy()
        metadata["cache_dir"] = manifest_path.parent
        return metadata

    def _load_module(self):
        module = make_lightning_module(self.config)
        module = module.__class__.load_from_checkpoint(
            self.checkpoint,
            network=self.config.network,
            optimizer=self.config.optimizer,
            lr_scheduler=self.config.lr_scheduler,
        )
        module.eval()
        return module

    def _build_dataloaders(self):
        print("Creating dataloaders for each condition from cached data.")
        dataloaders = []
        cached_datasets: dict[Path, CachedWindowDataset] = {}
        transforms = self._build_transforms(split="test")
        for group_values, group_df in tqdm(self.groupby, desc="Groups"):
            subset_datasets = []
            for cache_dir_str, df_cache in group_df.groupby("cache_dir"):
                cache_dir = Path(cache_dir_str)
                dataset = cached_datasets.get(cache_dir)
                if dataset is None:
                    dataset = CachedWindowDataset(cache_dir, transform=transforms)
                    cached_datasets[cache_dir] = dataset
                indices = df_cache["global_index"].astype(int).tolist()
                subset_datasets.append(Subset(dataset, indices))

            if not subset_datasets:
                continue

            if len(subset_datasets) == 1:
                group_dataset = subset_datasets[0]
            else:
                group_dataset = ConcatDataset(subset_datasets)

            dataloaders.append(
                DataLoader(
                    group_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                )
            )
        return dataloaders

    def _build_transforms(self, split: str):
        transform_configs = self.config.transforms.get(split)
        if transform_configs is None:
            return None
        transforms = [instantiate(cfg) for cfg in transform_configs]
        if not transforms:
            return None
        return Compose(transforms)

    def evaluate(self) -> pd.DataFrame:
        trainer = pl.Trainer(**self.config.trainer)
        dataloaders = self._build_dataloaders()
        results = trainer.test(self.module, dataloaders=dataloaders, verbose=True)
        return self._create_results_df(results)

    def _create_results_df(self, results):
        records = []
        for (group_values, _), metrics in zip(self.groupby, results):
            record = dict(zip(self.conditions, group_values))
            clean_metrics = {k.split("/")[0]: v for k, v in metrics.items()}
            record.update(clean_metrics)
            records.append(record)
        return pd.DataFrame(records)


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def cli(config: DictConfig):
    cache_root = Path(config.datamodule.cache_root).expanduser()
    evaluation = CachedEMG2PoseEvaluation(
        config=config,
        checkpoint=config.checkpoint,
        cache_root=cache_root,
        conditions=["generalization"],
        split="test",
        batch_size=config.batch_size,
    )
    results_df = evaluation.evaluate()

    results_path = Path(os.getcwd()).joinpath("results.csv")
    print(f"Saving results to {results_path}")
    results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    cli()
