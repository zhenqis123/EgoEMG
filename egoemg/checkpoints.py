"""Checkpoint formats and strict branch restoration shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def extract_state_dict(
    checkpoint: Any,
    *,
    keys: tuple[str, ...] = ("state_dict", "model_state_dict"),
) -> dict[str, torch.Tensor]:
    """Extract wrapped or raw weights; reject unsupported containers."""
    if isinstance(checkpoint, dict):
        for key in keys:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError("Unsupported checkpoint format")


def load_state_dict_from_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """Read trusted training weights, retaining model_state_dict precedence."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return extract_state_dict(checkpoint, keys=("model_state_dict", "state_dict"))


def load_branch(path):
    """Trusted local training checkpoint, strict full model restoration."""
    from hydra.utils import instantiate

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    conf = checkpoint["hyper_parameters"]["module_conf"]
    from omegaconf import OmegaConf

    if not OmegaConf.is_config(conf):
        conf = OmegaConf.create(conf)
    conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    if "pretrained" in conf:
        conf.pretrained = False  # Everything comes from the trained checkpoint.
    vision_conf = conf.get("vision_config")
    if OmegaConf.is_dict(vision_conf) and "pretrained" in vision_conf:
        vision_conf.pretrained = False
    model = instantiate(conf, _convert_="all")
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model
