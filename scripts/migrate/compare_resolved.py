#!/usr/bin/env python
"""Migrate validation: snapshot & diff resolved Hydra configs per experiment.

Usage:
  # Snapshot current (pre-migration) state of all experiments
  python scripts/migrate/compare_resolved.py snapshot --out /tmp/cfg_before.json

  # After migrating experiments, diff against the snapshot
  python scripts/migrate/compare_resolved.py diff --baseline /tmp/cfg_before.json

  # Or snapshot a single experiment for debugging
  python scripts/migrate/compare_resolved.py snapshot-one experiment=emgformer/regression_egoemg

This composes each experiment the same way `python -m emg2pose.train` does
(config_name=base, with the experiment override), resolves all interpolations,
then flattens to {key: value} for stable diffing.

Only training-relevant keys are compared (data/module/optimizer/trainer/loss/aug).
Cosmetic keys (hydra runtime dir, logger name, filename patterns) are excluded
so that migration deltas (which intentionally change filenames/logger names)
don't show as regressions.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

# Experiment groups to validate (matches the three main lines).
EXPERIMENT_GROUPS = ["emgformer", "fusion", "emg2pose"]

# Keys excluded from diff: cosmetic / runtime-only / intentionally-changed-by-migration.
# Nested keys matched by prefix.
EXCLUDE_PREFIXES = (
    "hydra.",          # runtime paths, job name
    "logger.",         # save_dir / name / version change intentionally
    "callbacks",       # filename patterns change intentionally
    "trainer.devices",  # machine-specific, not a model property
    # Interpolation-only keys that don't survive resolve (resolved value is the source)
)


def _flatten(cfg, prefix=""):
    """Flatten resolved config (plain dict from OmegaConf.to_container) to
    {dotted.key: value_str}. Dicts recurse; small lists serialize compactly;
    large lists collapse to a length marker."""
    out = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            elif isinstance(v, list):
                if len(v) <= 8:
                    out[key] = json.dumps(v, default=str)
                else:
                    out[key] = f"<list len {len(v)}>"
            else:
                out[key] = str(v)
    return out


def _filter_relevant(flat: dict) -> dict:
    """Keep only training-relevant keys, drop cosmetic/hydra internals."""
    return {k: v for k, v in flat.items()
            if not any(k.startswith(p) for p in EXCLUDE_PREFIXES)}


def _list_experiments() -> list[str]:
    """Return all experiment overrides like 'emgformer/regression_egoemg'."""
    exps = []
    for group in EXPERIMENT_GROUPS:
        for yml in sorted(glob.glob(str(CONFIG_DIR / "experiment" / group / "*.yaml"))):
            name = Path(yml).stem
            # skip _archive contents if any leak to top level
            if "_archive" in yml:
                continue
            exps.append(f"{group}/{name}")
    return exps


def snapshot_one(experiment: str) -> dict:
    """Compose & flatten a single experiment. Raises on hydra error."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="base",
                      overrides=[f"experiment={experiment}", "train=false", "eval=false"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    flat = _flatten(resolved)
    return _filter_relevant(flat)


def snapshot_all() -> dict:
    """Snapshot all experiments. Returns {experiment: {key: value}} and {experiment: error}."""
    out = {}
    errors = {}
    for exp in _list_experiments():
        try:
            out[exp] = snapshot_one(exp)
        except Exception as e:
            errors[exp] = f"{type(e).__name__}: {str(e)[:200]}"
    return {"experiments": out, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshot", help="Snapshot all experiments to JSON")
    p_snap.add_argument("--out", required=True)

    p_diff = sub.add_parser("diff", help="Diff current state against baseline")
    p_diff.add_argument("--baseline", required=True)

    p_one = sub.add_parser("snapshot-one", help="Snapshot a single experiment")
    p_one.add_argument("experiment", help="e.g. emgformer/regression_egoemg")

    args = parser.parse_args()

    if args.cmd == "snapshot":
        data = snapshot_all()
        Path(args.out).write_text(json.dumps(data, indent=2, default=str))
        n_ok = len(data["experiments"])
        n_err = len(data["errors"])
        print(f"Snapshot: {n_ok} OK, {n_err} errors -> {args.out}")
        if n_err:
            print("Errors:")
            for e, msg in data["errors"].items():
                print(f"  {e}: {msg}")
        sys.exit(0 if n_err == 0 else 1)

    elif args.cmd == "diff":
        baseline = json.loads(Path(args.baseline).read_text())
        current = snapshot_all()
        # Report experiments that failed to compose now (regression!)
        baseline_set = set(baseline["experiments"])
        current_set = set(current["experiments"])
        new_broken = current_set - baseline_set if baseline_set <= current_set else set()
        now_errors = set(current["errors"])
        baseline_errors = set(baseline["errors"])
        new_failures = now_errors - baseline_errors
        if new_failures:
            print(f"REGRESSION — {len(new_failures)} experiments now fail to compose:")
            for e in sorted(new_failures):
                print(f"  {e}: {current['errors'][e]}")
        # Diff each experiment's resolved config
        all_keys = baseline_set | current_set
        total_diffs = 0
        for exp in sorted(all_keys):
            b = baseline["experiments"].get(exp, {})
            c = current["experiments"].get(exp, {})
            all_k = sorted(set(b) | set(c))
            diffs = [(k, b.get(k), c.get(k)) for k in all_k if b.get(k) != c.get(k)]
            if diffs:
                total_diffs += len(diffs)
                print(f"\n{exp} ({len(diffs)} diffs):")
                for k, bv, cv in diffs[:20]:
                    print(f"  {k}: {bv} -> {cv}")
                if len(diffs) > 20:
                    print(f"  ... +{len(diffs)-20} more")
        if total_diffs == 0 and not new_failures:
            print(f"OK — all {len(all_keys)} experiments resolve identically to baseline.")
        else:
            print(f"\nTotal: {total_diffs} param diffs across experiments.")
        sys.exit(0 if (total_diffs == 0 and not new_failures) else 1)

    elif args.cmd == "snapshot-one":
        data = snapshot_one(args.experiment)
        print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
