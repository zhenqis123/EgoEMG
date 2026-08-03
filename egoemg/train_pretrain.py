# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EMGFormer pretraining entrypoint (multi-task SSL / supervised pretrain).

Shared training infrastructure (seed, callbacks, logger, trainer, fit/eval)
lives in `egoemg._train_utils`; this module only carries the pretrain-
specific module construction and Lightning-native checkpoint resume.
"""
import logging
from collections.abc import Callable, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from egoemg._train_utils import (
    build_callbacks,
    build_logger,
    build_trainer,
    run_train_eval,
    setup_runtime,
)
from egoemg.datamodule import make_data_module
from egoemg.lightning_pretrain import EmgPretrainModule

log = logging.getLogger(__name__)


def make_lightning_module(config: DictConfig):
    """Create the pretrain lightning module from experiment config."""
    return EmgPretrainModule(
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        gesture_spaces=config.gesture_spaces,
        loss_weights=config.loss_weights,
        mask_conf=config.masking,
        recon_loss=config.get("recon_loss", "mse"),
        angle_loss=config.get("angle_loss", "mae"),
        label_smoothing=config.get("label_smoothing", 0.0),
        datamodule=config.datamodule,
    )


def train(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
):
    """Pretrain train + eval entrypoint."""
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # ── Module construction (pretrain-specific) ──────────────────────────────
    if config.checkpoint is not None:
        log.info(f"Loading from checkpoint {config.checkpoint}")
        module = EmgPretrainModule.load_from_checkpoint(
            config.checkpoint,
            module_conf=config.module,
            optimizer_conf=config.optimizer,
            lr_scheduler_conf=config.lr_scheduler,
            gesture_spaces=config.gesture_spaces,
            loss_weights=config.loss_weights,
            mask_conf=config.masking,
            recon_loss=config.get("recon_loss", "mse"),
            angle_loss=config.get("angle_loss", "mae"),
            label_smoothing=config.get("label_smoothing", 0.0),
            datamodule=config.datamodule,
        )
    else:
        module = make_lightning_module(config)

    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = make_data_module(config)

    # ── Shared training infrastructure ───────────────────────────────────────
    setup_runtime(config)
    callbacks = build_callbacks(config, extra_callbacks, ensure_model_summary=False)
    logger = build_logger(config)
    trainer = build_trainer(config, callbacks, logger)

    # reload_best=False: pretrain keeps the last checkpoint (no best-val reload).
    return run_train_eval(trainer, module, datamodule, config, reload_best=False)


@hydra.main(config_path="../config", config_name="pretrain", version_base="1.1")
def cli(config: DictConfig):
    train(config)


if __name__ == "__main__":
    cli()
