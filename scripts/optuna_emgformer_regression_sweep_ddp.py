#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import optuna
from optuna.trial import TrialState


def _parse_devices(devices: str) -> list[int]:
    return [int(item) for item in devices.split(",") if item.strip() != ""]


def _write_trial_params(
    trial: optuna.Trial, params_path: Path
) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "module.decoder.dropout": trial.suggest_float(
            "module.decoder.dropout", 0.05, 0.25
        ),
        "transforms.train.3.mask_prob": trial.suggest_float(
            "transforms.train.3.mask_prob", 0.2, 0.7
        ),
        "transforms.train.4.num_masks": trial.suggest_categorical(
            "transforms.train.4.num_masks", [0, 3, 5, 7]
        ),
        "transforms.train.5.num_masks": trial.suggest_categorical(
            "transforms.train.5.num_masks", [0, 3, 5, 7]
        ),
        "transforms.train.6.apply_prob": trial.suggest_float(
            "transforms.train.6.apply_prob", 0.2, 0.7
        ),
    }
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(params, handle)
    return params


def _read_new_metrics(metrics_path: Path, offset: int) -> tuple[list[dict], int]:
    if not metrics_path.exists():
        return [], offset
    with metrics_path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        new_offset = handle.tell()
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries, new_offset


def _run_trial(
    trial: optuna.Trial,
    args: argparse.Namespace,
    devices: list[int],
    study: optuna.Study,
) -> None:
    trial_dir = Path(args.work_dir) / f"trial_{trial.number}"
    params_path = trial_dir / "params.json"
    metrics_path = trial_dir / "metrics.jsonl"
    result_path = trial_dir / "result.json"
    _write_trial_params(trial, params_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(len(devices)),
        "scripts/run_emgformer_trial_ddp.py",
        "--trial-params",
        str(params_path),
        "--metrics-path",
        str(metrics_path),
        "--result-path",
        str(result_path),
        "--data-split",
        args.data_split,
        "--max-epochs",
        str(args.max_epochs),
        "--seed",
        str(args.seed),
        "--num-devices",
        str(len(devices)),
        "--strategy",
        "ddp",
    ]
    if args.data_location:
        cmd.extend(["--data-location", args.data_location])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])

    proc = subprocess.Popen(cmd, env=env)
    offset = 0
    pruned = False
    try:
        while proc.poll() is None:
            entries, offset = _read_new_metrics(metrics_path, offset)
            for entry in entries:
                if "metric" not in entry or "epoch" not in entry:
                    continue
                trial.report(float(entry["metric"]), step=int(entry["epoch"]))
                if trial.should_prune():
                    pruned = True
                    proc.terminate()
                    break
            if pruned:
                break
            time.sleep(args.poll_interval)
    finally:
        if pruned and proc.poll() is None:
            proc.terminate()
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    if pruned:
        study.tell(trial, state=TrialState.PRUNED)
        return

    if proc.returncode != 0:
        study.tell(trial, state=TrialState.FAIL)
        return

    if not result_path.exists():
        study.tell(trial, state=TrialState.FAIL)
        return

    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    study.tell(trial, float(result["metric"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--study-name", type=str, default="emgformer_regression_small")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--data-split", type=str, default="mini_split")
    parser.add_argument("--data-location", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--work-dir", type=str, default="optuna_ddp_runs")
    args = parser.parse_args()

    devices = _parse_devices(args.devices)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    storage = optuna.storages.RDBStorage(args.storage) if args.storage else None

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage if args.storage else None,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True if args.storage else False,
    )

    for _ in range(args.n_trials):
        trial = study.ask()
        _run_trial(trial, args, devices, study)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
