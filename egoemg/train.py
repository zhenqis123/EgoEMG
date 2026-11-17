# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Supervised training entrypoint for EMG→pose, vision→pose, and fusion.

Shared training infrastructure (seed, callbacks, logger, trainer, fit/eval)
lives in `egoemg.training.runtime`; this module only carries the supervised-
specific module construction and checkpoint-weight loading.
"""

from egoemg.checkpoints import extract_state_dict
import logging
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    category=UserWarning,
)

from egoemg.training.runtime import (
    build_callbacks,
    build_logger,
    build_trainer,
    run_train_eval,
    setup_runtime,
)
from egoemg.training.datamodule import make_data_module
from egoemg.training.prediction import EmgPredictionModule

log = logging.getLogger(__name__)


def make_lightning_module(config: DictConfig):
    """Create lightning module from experiment config."""
    module_target = str(config.module.get("_target_", ""))
    if module_target == "egoemg.models.modules.emgformer_pretrain.EmgformerPretrain":
        raise ValueError(
            "This experiment config targets EmgformerPretrain and must be run via "
            "`python -m egoemg.train_pretrain ...`, not `python -m egoemg.train ...`."
        )
    module_class = EmgPredictionModule
    prediction_type = config.get("prediction_type", "auto")
    if prediction_type not in {"auto", "emg", "vision", "residual_fusion"}:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")
    if prediction_type == "residual_fusion" or (
        prediction_type == "auto" and module_target in {
            "egoemg.models.modules.bimanual_simple_residual.BimanualSimpleResidualFusion",
            "egoemg.models.modules.bimanual_direct_fusion.BimanualDirectFusion",
        }
    ):
        from egoemg.training.vision import ResidualFusionPredictionModule

        module_class = ResidualFusionPredictionModule
    elif prediction_type == "vision" or (
        prediction_type == "auto" and module_target in {
            "egoemg.models.modules.bimanual_vision.BimanualVisionViTPose",
            "egoemg.models.modules.bimanual_vision.BimanualVisionResNetPose",
            "egoemg.models.modules.bimanual_temporal_vision.BimanualTemporalVisionPose",
        }
    ):
        from egoemg.training.vision import VisionPredictionModule

        module_class = VisionPredictionModule
    module_conf = config.module
    if config.get("self_contained_checkpoint", False):
        module_conf = OmegaConf.create(
            OmegaConf.to_container(module_conf, resolve=True)
        )
    return module_class(
        module_conf=module_conf,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        loss_weights=config.loss_weights,
        pretrained_checkpoint=config.get("pretrained_checkpoint"),
        pretrained_strict=config.get("pretrained_strict", False),
        freeze_backbone=config.get("freeze_backbone", False),
        pretrained_emg_checkpoint=config.get("pretrained_emg_checkpoint"),
        stage2_vision_checkpoint=config.get("stage2_vision_checkpoint"),
        component_lr_scales=config.get("component_lr_scales"),
        ignore_head_tail_dims=config.get("ignore_head_tail_dims", 0),
        datamodule=config.datamodule,
        batch_augmentation=config.get("batch_augmentation"),
        val_episode_name_mapping=config.get("val_episode_name_mapping"),
        anchor_loss_weight=config.get("anchor_loss_weight", 0.0),
        anchor_shuffle_fraction=config.get("anchor_shuffle_fraction", 0.0),
        train_monitor_interval=config.get("train_monitor_interval", 1),
        enable_fingertip_loss=config.get("enable_fingertip_loss", True),
        **({"validation_user_macro": True}
           if config.get("validation_user_macro", False) else {}),
    )


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor] | None:
    """Training treats unsupported checkpoint containers as absent weights."""
    try:
        return extract_state_dict(checkpoint)
    except ValueError:
        return None


def _is_pretrain_checkpoint(state_dict: dict[str, torch.Tensor] | None) -> bool:
    """Heuristic: detect pretrain-format checkpoints by their head prefixes."""
    if not state_dict:
        return False
    pretrain_prefixes = (
        "model.recon_head.",
        "model.gesture_head.",
        "model.keystroke_head.",
        "model.angle_head.",  # Pretrain uses angle_head, regular uses head
        "model.mask_embedding",
        "model.projection.",
    )
    return any(key.startswith(pretrain_prefixes) for key in state_dict.keys())


def _load_from_checkpoint(config: DictConfig):
    """Build module then load backbone + head weights from a checkpoint file."""
    log.info(f"Loading from checkpoint {config.checkpoint}")
    ckpt_path = Path(config.checkpoint).expanduser()
    # Checkpoints record OmegaConf hyperparameters that PyTorch 2.6+'s
    # weights-only default cannot deserialize (trusted local artifact).
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    strict = bool(config.get("pretrained_strict", False))

    if _is_pretrain_checkpoint(state_dict):
        log.info("Detected pretrain checkpoint. Loading backbone and angle head.")
    else:
        log.info(
            "Detected regular checkpoint. Loading backbone and matching head weights."
        )

    module = make_lightning_module(config)
    # NOTE: _load_pretrained_backbone re-loads the file internally; future cleanup
    # (checkpoint_utils) should pass the already-loaded state_dict to avoid double I/O.
    module._load_pretrained_backbone(str(ckpt_path), strict=strict)
    module._load_pretrained_angle_head(str(ckpt_path), strict=strict)
    return module


def train(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
):
    """Supervised train + eval entrypoint."""
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")
    # Seed before model/head initialization so fresh grid arms share initial
    # weights for a fixed architecture, not only the later data RNG state.
    setup_runtime(config)

    # ── Module construction (supervised-specific) ────────────────────────────
    if config.checkpoint is not None:
        module = _load_from_checkpoint(config)
    else:
        log.info(f"Instantiating LightningModule {EmgPredictionModule}")
        module = make_lightning_module(config)

    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = make_data_module(config)

    # ── Shared training infrastructure ───────────────────────────────────────
    callbacks = build_callbacks(config, extra_callbacks, ensure_model_summary=True)
    logger = build_logger(config)
    trainer = build_trainer(config, callbacks, logger)

    return run_train_eval(trainer, module, datamodule, config, reload_best=True)


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def cli(config: DictConfig):
    train(config)


if __name__ == "__main__":
    cli()
