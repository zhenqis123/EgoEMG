from pathlib import Path

from hydra import compose, initialize_config_dir


def test_every_active_experiment_config_composes() -> None:
    """Archive configs are historical; every active recipe must resolve in Hydra."""
    repo_root = Path(__file__).resolve().parents[2]
    config_root = repo_root / "config"
    experiment_root = config_root / "experiment"
    configs = sorted(
        path for path in experiment_root.rglob("*.yaml") if "_archive" not in path.parts
    )
    assert configs, "expected at least one active experiment config"

    with initialize_config_dir(version_base="1.1", config_dir=str(config_root)):
        for path in configs:
            name = path.relative_to(experiment_root).with_suffix("").as_posix()
            config = compose(config_name="base", overrides=[f"experiment={name}"])
            assert config.get("trainer") is not None, name
