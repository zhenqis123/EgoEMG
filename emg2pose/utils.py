# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from collections.abc import Iterator
from typing import Any

from joblib import Parallel
import numpy as np

import pandas as pd

from hydra import compose, initialize, initialize_config_dir
from scipy.interpolate import interp1d

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm


def instantiate_optimizer_and_scheduler(
    params: Iterator[nn.Parameter],
    optimizer_config: DictConfig,
    lr_scheduler_config: DictConfig | None,
) -> dict[str, Any]:
    optimizer = instantiate(optimizer_config, params)
    out = {"optimizer": optimizer}

    if lr_scheduler_config is not None:
        scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)
        lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
        out["lr_scheduler"] = OmegaConf.to_container(lr_scheduler)
    return out


def generate_hydra_config_from_overrides(
    config_path: str = "../config",
    version_base: str | None = None,
    config_name: str = "base",
    overrides: list[str] | None = None,
) -> DictConfig:

    if overrides is None:
        overrides = []

    if os.path.isabs(config_path):
        with initialize_config_dir(config_dir=config_path, version_base=version_base):
            config = compose(config_name=config_name, overrides=overrides)
    else:
        with initialize(config_path=config_path, version_base=version_base):
            config = compose(config_name=config_name, overrides=overrides)

    return config


def load_splits(
    metadata_file: str,
    subsample: float = 1.0,
    random_seed: int = 0,
) -> dict[str, list[str]]:
    """Load train, val, and test datasets from metadata csv."""

    # Load dataframe
    df = pd.read_csv(metadata_file)

    # Optionally subsample
    df = df.groupby("split").apply(
        lambda x: x.sample(frac=subsample, random_state=random_seed)
    )
    df.reset_index(drop=True, inplace=True)

    # Format as dictionary
    splits = {}
    for split, df_ in df.groupby("split"):
        splits[split] = list(df_.filename)

    return splits


def get_contiguous_ones(binary_vector: np.ndarray) -> list[tuple[int, int]]:
    """Get a list of (start_idx, end_idx) for each contiguous block of True values."""
    if (binary_vector == 0).all():
        return []

    ones = np.where(binary_vector)[0]
    boundaries = np.where(np.diff(ones) != 1)[0]
    return [
        (ones[i], ones[j])
        for i, j in zip(
            np.insert(boundaries + 1, 0, 0), np.append(boundaries, len(ones) - 1)
        )
    ]


def get_ik_failures_mask(joint_angles: np.ndarray) -> np.ndarray:
    """Compute mask that is True where there are no ik failures."""
    zeros = np.zeros_like(joint_angles)  # (..., joint)
    is_zero = np.isclose(joint_angles, zeros)
    return ~np.all(is_zero, axis=-1)


def downsample(
    array: np.ndarray, native_fs: int = 2000, target_fs: int = 30
) -> np.ndarray:
    """
    Downsamples a given array from its native sampling frequency to a target
    sampling frequency.

    Args:
        array (np.ndarray): The input array to be downsampled.
        native_fs (int, optional): The native sampling frequency of the input array.
        Defaults to 2000.
        target_fs (int, optional): The target sampling frequency. Defaults to 30.

    Returns:
        np.ndarray: The downsampled array.
    """

    # Create a time array for the original signal
    t_native = np.arange(array.shape[0]) / native_fs

    # Calculate the number of samples in the downsampled signal
    num_samples = int(array.shape[0] * target_fs / native_fs)

    # Create a time array for the downsampled signal
    t_target = np.linspace(0, t_native[-1], num_samples)

    # Interpolate the original signal at the downsampled time points
    f = interp1d(
        t_native, array, axis=0, kind="linear", fill_value=np.nan, bounds_error=False
    )
    downsampled_array = f(t_target)

    return downsampled_array


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, tqdm_kwargs=None, *args, **kwargs):
        """Joblib Parallel with a tqdm progress bar.

        See: https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib

        Parameters
        ----------
        use_tqdm : bool
            Whether to show the progress bar. Default is True.
        total : int or None
            Total number of tasks to anchor the progress bar size. If None, the
            total will be inferred from the number of completed tasks.
        tqdm_kwargs : Dict[str, Any]
            Additional keyword arguments to pass to `tqdm` when creating the progress
            bar.
        Other parameters are passed to `joblib.Parallel`.
        """
        self._use_tqdm = use_tqdm
        self._total = total
        self._tqdm_kwargs = tqdm_kwargs if tqdm_kwargs is not None else {}
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm, total=self._total, **self._tqdm_kwargs
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
