"""Output-path behavior for current offline evaluation."""

from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from egoemg.test_analysis import _results_output_path


@pytest.mark.parametrize("with_hydra", [False, True])
def test_results_stay_in_evaluation_or_run_directory(tmp_path, monkeypatch, with_hydra):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "logs" / "evaluation" / "selected_run"
    hydra = OmegaConf.create({"hydra": {"runtime": {"output_dir": str(run_dir)}}})
    monkeypatch.setattr(HydraConfig.instance(), "cfg", hydra if with_hydra else None)
    expected = run_dir if with_hydra else tmp_path / "logs/evaluation/standalone"
    assert Path(_results_output_path()) == expected / "results.csv"
    assert expected.is_dir()
    assert not (tmp_path / "results.csv").exists()
