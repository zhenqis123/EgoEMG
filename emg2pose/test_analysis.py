# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass

import hydra
import pandas as pd
import pytorch_lightning as pl
from emg2pose.datasets.multisession_emg2pose_dataset import (
    MultiSessionWindowedEmgDataset,
)

from emg2pose.train import make_lightning_module
from emg2pose.transforms import Compose
from hydra.utils import instantiate

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_DATA_DIR = "/emg2pose_data/"


@dataclass
class EMG2PoseEvaluation:
    """
    Run offline evaluation for a trained emg2pose model.

    Metrics are computed for each combination of conditions, e.g. for each
    (generalization, user) or each (generalization, user, stage).
    """

    config: DictConfig
    checkpoint: str
    conditions: list[str]
    window_length: int = 10_000
    split: str = "test"
    skip_ik_failures: bool = True
    batch_size: int = 512

    def __post_init__(self):
        self.data_dir = self.config.data_location
        self.df = self.get_corpus_df()
        self.groupby = self.df.groupby(self.conditions)
        self.module = self.get_module()
        self.dataloaders = self.get_dataloaders()
        self.batch_size = self.config.get("batch_size", self.batch_size)

    def get_corpus_df(self):
        metadata_file = os.path.join(self.data_dir, "metadata.csv")
        df = pd.read_csv(metadata_file).query(f"split=='{self.split}'")

        # Optionally subsample corpus for testing purposes
        corpus_subsample = self.config.get("corpus_subsample", 1)
        if corpus_subsample < 1:
            print(f"Subsampling corpus by {corpus_subsample}")
            df = df.sample(frac=corpus_subsample, random_state=0)

        return df

    def get_module(self):
        module = make_lightning_module(self.config)
        # import torch
        # sd = torch.load("/home/xiziheng/develop/emg2pose/test.pth", map_location="cpu")
        # missing, unexpected = module.load_state_dict(sd, strict=False)
        # print("missing:", len(missing))
        # print("unexpected:", len(unexpected))
        module = module.__class__.load_from_checkpoint(
            self.config.checkpoint,
            module_conf=self.config.module,
            optimizer_conf=self.config.optimizer,
            lr_scheduler_conf=self.config.lr_scheduler,
            loss_weights=self.config.loss_weights,
        )
        module.eval()
        return module

    def get_dataloaders(self) -> list[DataLoader]:
        """
        Get list of dataloaders, each corresponding to a single groupby condition
        (e.g., [user, stage]).
        """

        transforms = Compose(instantiate(self.config.transforms[self.split], _convert_="all"))
        context_length = self.module.model.left_context + self.module.model.right_context
        effective_window_length = self.window_length + context_length
        stride = self.window_length
        max_open_files = int(self.config.datamodule.get("max_open_files", 32))

        print("Creating dataloaders for each condition.")
        dataloaders = []
        for _, df_ in tqdm(self.groupby):
            session_paths = [
                os.path.join(self.data_dir, f"{name}.hdf5") for name in df_.filename
            ]
            dataset = MultiSessionWindowedEmgDataset(
                hdf5_paths=session_paths,
                transform=transforms,
                window_length=effective_window_length,
                stride=stride,
                padding=(0, 0),
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
                max_open_files=max_open_files,
            )
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=12, pin_memory=True
            )
            dataloaders.append(dataloader)

        return dataloaders

    def create_results_df(self, results: list[dict[str, float]]):
        """Convert results list to a dataframe with all condition information."""
        records = []
        for (vals, _), result in zip(self.groupby, results):
            record = dict(zip(self.conditions, vals))
            result = {k.split("/")[0]: v for k, v in result.items()}
            record.update(result)
            records.append(record)
        return pd.DataFrame(records)

    def evaluate(self) -> pd.DataFrame:
        """Run analysis for split."""
        trainer = pl.Trainer(**self.config.trainer)
        results = trainer.test(self.module, dataloaders=self.dataloaders, verbose=True)
        results_df = self.create_results_df(results)
        return results_df


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def cli(config: DictConfig):
    evaluation = EMG2PoseEvaluation(
        config=config,
        checkpoint=config.checkpoint,
        conditions=["generalization"],
        window_length=10_000,
        split="test",
        skip_ik_failures=True,
    )
    results_df = evaluation.evaluate()

    # Save results to a csv in the logs folder
    results_filename = os.path.join(os.getcwd(), "results.csv")
    print(f"Saving results to {results_filename}")
    results_df.to_csv(results_filename, index=False)


if __name__ == "__main__":
    cli()
