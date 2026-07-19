#!/usr/bin/env python
"""Post-migration cleanup: remove redundant overrides + dead fields.

After migrate_experiments.py, many experiments still carry `override /group: X`
entries whose value X equals what the lineage already selects for that group.
This script compares each experiment's override values against the lineage's
group selections and drops the redundant lines, so each experiment truly
expresses only deltas.

Also optional cleanups:
  --drop-field <dotted.key>   remove a top-level field from all migrated
                              experiments (e.g. wilor_checkpoint_path from
                              non-wilor fusion experiments).
  --limit-groups              only process experiments whose defaults inherit
                              the given lineage (default: all three).

Usage:
  python scripts/migrate/cleanup_experiments.py                # drop redundant overrides
  python scripts/migrate/cleanup_experiments.py --verify       # then verify all
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
BASELINE_SNAPSHOT = Path("/tmp/cfg_baseline.json")


def _lineage_for_experiment(exp_path: Path) -> str | None:
    """Return the lineage name an experiment inherits (via defaults or parent)."""
    txt = exp_path.read_text()
    # Direct: defaults has /lineage/<name>
    m = re.search(r"/lineage/(\w+)", txt)
    if m:
        return m.group(1)
    # Via parent experiment: look up the parent's lineage recursively.
    # Parse defaults for /experiment/<group>/<name>
    m = re.search(r"/experiment/(\w+)/(\w+)", txt)
    if m:
        parent = CONFIG / "experiment" / m.group(1) / f"{m.group(2)}.yaml"
        if parent.exists() and parent != exp_path:
            return _lineage_for_experiment(parent)
    return None


def _lineage_group_selections(lineage: str) -> dict[str, str]:
    """Parse a lineage file's defaults and return {group_path: selected_option}.

    e.g. {'/dataset': 'egoemg_angle_regression', '/module': 'emgformer', ...}
    Handles both 'override /group: opt' and '/group: opt' forms.
    """
    lp = CONFIG / "lineage" / f"{lineage}.yaml"
    if not lp.exists():
        return {}
    out = {}
    for line in lp.read_text().split("\n"):
        s = line.strip()
        if not s.startswith("-"):
            continue
        s = s.lstrip("-").strip()
        # forms: "/group: opt"  or  "override /group: opt"
        s2 = s[len("override "):].strip() if s.startswith("override ") else s
        if ":" in s2:
            grp, opt = s2.split(":", 1)
            out[grp.strip()] = opt.strip()
    return out


def _experiment_override_lines(text: str) -> list[tuple[int, str, str, str]]:
    """Return [(line_idx_0based, raw_line, group, option)] for each defaults
    entry of form 'override /group: opt' or '/group: opt'."""
    out = []
    in_defaults = False
    for idx, line in enumerate(text.split("\n")):
        s = line.strip()
        if s == "defaults:" or s.startswith("defaults:"):
            in_defaults = True
            continue
        if in_defaults:
            if s and not line[0].isspace() and not s.startswith("-"):
                break  # next top-level key
            if s.startswith("-"):
                body = s.lstrip("-").strip()
                is_override = body.startswith("override ")
                if is_override:
                    body = body[len("override "):].strip()
                if ":" in body:
                    grp, opt = body.split(":", 1)
                    out.append((idx, line, grp.strip(), opt.strip()))
    return out


def cleanup_redundant_overrides(exp_path: Path) -> tuple[bool, int]:
    """Drop override lines whose value matches the lineage's selection.
    Returns (changed, n_lines_dropped)."""
    lineage = _lineage_for_experiment(exp_path)
    if not lineage:
        return False, 0
    lin_sel = _lineage_group_selections(lineage)
    txt = exp_path.read_text()
    overrides = _experiment_override_lines(txt)
    drop_indices = set()
    for idx, raw, grp, opt in overrides:
        if grp in lin_sel and lin_sel[grp] == opt:
            drop_indices.add(idx)
    if not drop_indices:
        return False, 0
    lines = txt.split("\n")
    new_lines = [l for i, l in enumerate(lines) if i not in drop_indices]
    exp_path.write_text("\n".join(new_lines))
    return True, len(drop_indices)


def cleanup_field(exp_path: Path, field: str) -> bool:
    """Remove a top-level field (dotted key supported for one level)."""
    txt = exp_path.read_text()
    # match `field: ...` at top level (no leading whitespace)
    pat = re.compile(rf"^{re.escape(field)}:\s*.*\n", re.MULTILINE)
    new, n = pat.subn("", txt)
    if n:
        exp_path.write_text(new)
    return n > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="after cleanup, run verify-one on all baseline-OK experiments")
    args = parser.parse_args()

    # Find all migrated experiments (those whose defaults reference a lineage,
    # directly or via parent).
    migrated = []
    for yml in sorted(CONFIG.glob("experiment/*/*.yaml")):
        if "_archive" in str(yml):
            continue
        lin = _lineage_for_experiment(yml)
        if lin:
            migrated.append((yml, lin))
    print(f"Found {len(migrated)} migrated experiments")

    total_dropped = 0
    changed = 0
    for yml, lin in migrated:
        rel = yml.relative_to(REPO)
        did, n = cleanup_redundant_overrides(yml)
        if did:
            changed += 1
            total_dropped += n
            print(f"  {rel}: dropped {n} redundant override(s)")
    print(f"\nCleanup: {changed} files changed, {total_dropped} override lines dropped")

    if args.verify:
        print("\n=== Verifying all experiments ===")
        snap = json.loads(BASELINE_SNAPSHOT.read_text())
        ok = fail = 0
        for exp in sorted(snap["experiments"]):
            r = subprocess.run(
                [sys.executable, "scripts/migrate/compare_resolved.py",
                 "verify-one", exp],
                cwd=str(REPO), capture_output=True, text=True)
            out = r.stdout.strip()
            diffs = [l.strip().rstrip(":") for l in out.split("\n")
                     if l.startswith("  ") and not l.startswith("    ")
                     and ":" in l and "keys differ" not in l]
            non_lineage = [d for d in diffs if d != "lineage"]
            if not non_lineage:
                ok += 1
            else:
                fail += 1
                print(f"  ✗ {exp}: {non_lineage[:5]}")
        print(f"\nVerify: {ok} OK, {fail} fail")


if __name__ == "__main__":
    main()
