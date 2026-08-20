"""Config-level regression test: the multitask pretraining module must be
instantiable with real heads (not plain dicts) and must accept the 16-channel
pretraining corpus input.

Found by the 2026-08-20 model review: the experiment config overrode the
heads without `_target_` (Hydra returned dicts; first forward raised
TypeError) and the featurizer's first conv expected 8 channels while the
pretraining corpus is harmonized to 16.
"""
from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


CONFIG_DIR = str((Path(__file__).resolve().parents[2] / "config").resolve())


def _compose():
    with initialize_config_dir(CONFIG_DIR, version_base="1.1"):
        return compose(
            config_name="pretrain",
            overrides=["experiment=emgformer/pretrain_multitask"],
        )


def test_multitask_heads_are_modules():
    cfg = _compose()
    for head in ("recon_head", "angle_head", "gesture_head", "keystroke_head"):
        assert cfg.module[head].get("_target_"), f"{head} lacks _target_"
        module = instantiate(cfg.module[head])
        assert isinstance(module, torch.nn.Module), f"{head} did not instantiate to a Module"


def test_multitask_module_forward_16ch():
    cfg = _compose()
    model = instantiate(cfg.module)
    # T=2000 survives the TDS stack's ~80x temporal downsampling on CPU.
    emg = torch.randn(2, 16, 2000)
    outputs = model({"emg": emg})
    assert outputs["recon"].shape[:2] == (2, 16), outputs["recon"].shape
    assert outputs["angles"].shape[:2] == (2, 22)
    assert outputs["gesture_logits"].shape[1] == cfg.gesture_num_classes
    # keystroke logits are permuted to (T, B, C) for CTC loss.
    assert outputs["keystroke_logits"].shape[2] == 98
