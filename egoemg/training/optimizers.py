"""Instantiate optimizers and optional schedulers from Hydra configuration."""

from collections.abc import Iterator
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn


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
