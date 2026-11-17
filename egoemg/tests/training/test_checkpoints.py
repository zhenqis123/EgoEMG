from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from egoemg.checkpoints import extract_state_dict, load_state_dict_from_checkpoint
from egoemg.train import _extract_state_dict
from egoemg.training.runtime import run_train_eval


@pytest.mark.parametrize("key", [None, "state_dict", "model_state_dict"])
def test_common_checkpoint_formats(tmp_path, key):
    weights = {"model.weight": torch.tensor([1.0, 2.0])}
    checkpoint = weights if key is None else {key: weights, "epoch": 4}
    path = tmp_path / "weights.ckpt"
    torch.save(checkpoint, path)
    assert extract_state_dict(checkpoint) is weights
    assert _extract_state_dict(checkpoint) is weights
    torch.testing.assert_close(
        load_state_dict_from_checkpoint(path)["model.weight"], weights["model.weight"]
    )


def test_existing_format_precedence_is_explicit(tmp_path):
    state = {"weight": torch.tensor(1.0)}
    model_state = {"weight": torch.tensor(2.0)}
    checkpoint = dict(state_dict=state, model_state_dict=model_state)
    assert extract_state_dict(checkpoint) is state
    path = tmp_path / "weights.ckpt"
    torch.save(checkpoint, path)
    assert load_state_dict_from_checkpoint(path)["weight"].item() == 2.0


def test_invalid_container_and_missing_file_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        extract_state_dict([1, 2])
    assert _extract_state_dict([1, 2]) is None
    with pytest.raises(FileNotFoundError, match="Pretrained checkpoint not found"):
        load_state_dict_from_checkpoint(tmp_path / "missing.ckpt")


class _TinyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))


@pytest.mark.parametrize("has_best", [True, False])
@pytest.mark.parametrize("reload_best", [True, False])
def test_eval_uses_selected_weights_in_existing_module(
    tmp_path, monkeypatch, has_best, reload_best
):
    module = _TinyModule()
    best = tmp_path / "best.ckpt"
    last = tmp_path / "last.ckpt"
    # Include OmegaConf, as in real checkpoints (requires trusted full loading).
    for path, value in [(best, 2.0), (last, 3.0)]:
        torch.save(
            {
                "state_dict": {"weight": torch.tensor(value)},
                "hyper_parameters": OmegaConf.create({"example": True}),
            },
            path,
        )

    def disallow_reconstruction(*args, **kwargs):
        raise AssertionError("Evaluation must not construct another model")

    monkeypatch.setattr(_TinyModule, "load_from_checkpoint", disallow_reconstruction)
    expected = (2.0 if has_best else 3.0) if reload_best else 5.0
    evaluated = []

    def fit(model, data, ckpt_path):
        assert model is module
        with torch.no_grad():
            model.weight.fill_(5.0)

    def evaluate(model, data):
        assert model is module
        assert model.weight.item() == expected
        evaluated.append(model)
        return [{"mae": expected}]

    trainer = SimpleNamespace(
        fit=fit,
        validate=evaluate,
        test=evaluate,
        checkpoint_callback=SimpleNamespace(
            best_model_path=str(best) if has_best else "",
            last_model_path=str(last),
        ),
    )
    config = OmegaConf.create({"train": True, "eval": True})
    results = run_train_eval(trainer, module, None, config, reload_best=reload_best)
    assert len(evaluated) == 2
    key = "best_checkpoint" if has_best else "last_checkpoint"
    assert results[key] == str(best if has_best else last)


@pytest.mark.parametrize("config_name", ["base", "pretrain"])
def test_logger_and_trainer_share_hydra_run_directory(
    tmp_path, monkeypatch, config_name
):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    from egoemg.training.runtime import build_logger, build_trainer

    config_root = Path(__file__).resolve().parents[3] / "config"
    if not (config_root / f"{config_name}.yaml").exists():
        pytest.skip("Pretraining is outside the minimal release")
    with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
        config = compose(config_name=config_name, return_hydra_config=True)
    run_dir = tmp_path / "logs/training/selected_run"
    config.hydra.runtime.output_dir = str(run_dir)
    monkeypatch.setattr(HydraConfig.instance(), "cfg", None)
    HydraConfig.instance().set_config(config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    logger = build_logger(config)
    trainer = build_trainer(
        OmegaConf.create({"trainer": {"enable_model_summary": False}}), [], logger
    )
    assert Path(trainer.default_root_dir) == run_dir
    assert Path(trainer.log_dir) == run_dir
    logger.log_metrics({"smoke": 1.0}, step=0)
    logger.finalize("success")
    assert list(run_dir.glob("events.out.tfevents.*"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["logs"]
