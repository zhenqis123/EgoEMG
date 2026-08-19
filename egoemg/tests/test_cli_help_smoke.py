"""CLI entrypoint smoke tests (data-independent).

Verifies that the public entrypoints documented in the README parse their
arguments and print help without requiring datasets, checkpoints, or GPUs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, *args, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{args} --help failed ({result.returncode}):\n"
        f"{result.stdout.decode()[-2000:]}\n{result.stderr.decode()[-2000:]}"
    )


def test_visualize_dataset_help():
    _run(["scripts/viz/visualize_dataset.py"])


def test_train_entrypoint_help():
    _run(["-m", "egoemg.train"])
