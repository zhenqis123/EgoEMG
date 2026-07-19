#!/usr/bin/env python
"""Batch-migrate experiments to inherit lineage files (text-preserving).

Operates on the raw YAML text to preserve formatting and avoid Hydra-defaults
serialization pitfalls. For each experiment:

  1. Load baseline text from a git worktree.
  2. In the `defaults:` block, drop entries whose group the lineage already
     provides (root experiments only). Prepend `- /lineage/<lineage>` for root
     experiments; child experiments (inheriting another experiment) are left
     to inherit lineage through their parent.
  3. Drop top-level framework keys the lineage pins (precision,
     gradient_clip_val, matmul_precision, etc.) from the experiment body.
  4. Write the migrated text to the main worktree.
  5. Verify resolved-config equivalence against the baseline snapshot. The
     only acceptable diff is the new `lineage:` key.

Usage:
  python scripts/migrate/migrate_experiments.py emgformer
  python scripts/migrate/migrate_experiments.py fusion
  python scripts/migrate/migrate_experiments.py classic
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path("/tmp/emg2pose_baseline")
BASELINE_SNAPSHOT = Path("/tmp/cfg_baseline.json")

# Defaults group paths each lineage pins. Matching is robust to 'override '
# prefix: an entry like 'override /module/featurizer: tds_slim' matches group
# '/module/featurizer'.
LINEAGE_DEFAULTS_GROUPS = {
    "emgformer": {"/module", "/augmentation", "/module/featurizer",
                  "/dataset", "/datamodule", "/transforms", "/lr_scheduler"},
    "fusion": {"/dataset", "/datamodule", "/lr_scheduler"},
    "classic": {"/dataset", "/datamodule", "/transforms", "/lr_scheduler"},
}

# Framework top-level keys + trainer sub-keys the lineage pins. These lines
# get deleted from the experiment body (lineage provides them).
LINEAGE_FRAMEWORK_KEYS = {
    "emgformer": {"top": {"matmul_precision", "ignore_head_tail_dims"},
                  "trainer": {"gradient_clip_val", "precision", "check_val_every_n_epoch"}},
    "fusion": {"top": {"matmul_precision", "ignore_head_tail_dims"},
               "trainer": {"gradient_clip_val", "precision"}},
    "classic": {"top": {"matmul_precision", "ignore_head_tail_dims"},
                "trainer": {"max_epochs", "check_val_every_n_epoch", "log_every_n_steps"}},
}


def _entry_group(line: str) -> str | None:
    """From a defaults list line like '  - override /module/featurizer: tds_slim'
    extract the group path '/module/featurizer'."""
    s = line.strip().lstrip("-").strip()
    if not s or s == "_self_" or s.startswith("/lineage/") or s.startswith("/experiment/"):
        return None
    key = s.split(":", 1)[0].strip()
    if key.startswith("override "):
        key = key[len("override "):].strip()
    return key


def _is_defaults_entry(line: str) -> bool:
    """True if a text line is a defaults list entry (starts with optional
    whitespace then '- ')."""
    return bool(re.match(r"^\s*-\s", line))


def list_experiments(group: str) -> list[str]:
    snap = json.loads(BASELINE_SNAPSHOT.read_text())
    return sorted(e for e in snap["experiments"] if e.startswith(f"{group}/"))


def migrate_text(text: str, lineage: str) -> tuple[str, bool, bool]:
    """Migrate raw experiment text. Returns (new_text, is_root, changed).
    is_root = does not inherit another experiment (so lineage ref was added)."""
    lines = text.split("\n")
    # Locate defaults block.
    try:
        dflt_idx = next(i for i, l in enumerate(lines)
                        if l.strip() == "defaults:" or l.startswith("defaults:"))
    except StopIteration:
        return text, True, False

    # Collect defaults entries until next top-level key.
    entries = []  # list of (line_index, raw_line)
    i = dflt_idx + 1
    while i < len(lines):
        l = lines[i]
        if l and not l[0].isspace() and not l.lstrip().startswith("-"):
            break  # next top-level key
        if _is_defaults_entry(l):
            entries.append((i, l))
        i += 1

    # Detect inheritance.
    inherits = any("/experiment/" in l for _, l in entries)
    drop_groups = LINEAGE_DEFAULTS_GROUPS[lineage]

    # Build new defaults entries.
    if inherits:
        # Child: keep all entries verbatim (each override is intentional).
        new_entries = list(entries)
        is_root = False
    else:
        # Root: drop entries whose group the lineage already provides AND that
        # are not overrides (an experiment override of a lineage group is an
        # intentional delta, e.g. using aug_best instead of the lineage's
        # aug_extended — keep it so it wins over the lineage default).
        def _keep_root_entry(line: str) -> bool:
            grp = _entry_group(line)
            if grp is None:
                return True  # _self_, /lineage, /experiment
            if grp not in drop_groups:
                return True  # group not provided by lineage
            # Group IS provided by lineage. Keep only if this is an override
            # (intentional different choice).
            s = line.strip().lstrip("-").strip()
            return s.startswith("override ")

        new_entries = [(idx, l) for idx, l in entries if _keep_root_entry(l)]
        # Prepend lineage ref at the position of the first kept entry (or right
        # after 'defaults:').
        insert_at = new_entries[0][0] if new_entries else dflt_idx + 1
        indent = "  "  # standard 2-space indent for defaults entries
        lineage_line = f"{indent}- /lineage/{lineage}"
        # Rebuild: keep original first-entry position, insert lineage before it.
        new_entries_with_lineage = [(insert_at, lineage_line)] + new_entries
        new_entries = new_entries_with_lineage
        is_root = True

    # Rewrite defaults block lines.
    drop_idx = {idx for idx, _ in entries}  # original entry line indices
    keep_lines = [l for idx, l in enumerate(lines) if idx not in drop_idx]
    # For kept entries, we need to re-insert them. Simpler: rebuild the whole
    # defaults region.
    # Strategy: replace lines[dflt_idx+1 .. i-1] with new entry texts.
    head = lines[:dflt_idx + 1]
    tail = lines[i:]
    entry_texts = [l for _, l in new_entries]
    # Ensure _self_ is last: drop any existing _self_ then append one.
    entry_texts = [e for e in entry_texts if "_self_" not in e]
    entry_texts.append("  - _self_")
    new_lines = head + entry_texts + [""] + tail
    new_text = "\n".join(new_lines)

    # Drop framework keys from body (top-level + trainer block).
    fw = LINEAGE_FRAMEWORK_KEYS[lineage]
    body_lines = new_text.split("\n")
    out_lines = []
    in_trainer = False
    trainer_block_start = -1
    trainer_block_lines = []
    for l in body_lines:
        stripped = l.lstrip()
        indent_len = len(l) - len(stripped)
        # Detect trainer: block.
        if stripped.startswith("trainer:") and indent_len == 0:
            in_trainer = True
            trainer_block_start = len(out_lines)
            trainer_block_lines = []
            continue  # don't emit yet; decide after block ends
        if in_trainer:
            # Exit trainer block on a non-blank, non-comment, top-level line.
            if stripped and not stripped.startswith("#") and indent_len == 0:
                # exited trainer block: flush kept trainer lines (if any).
                if trainer_block_lines:
                    out_lines.append("trainer:")
                    out_lines.extend(trainer_block_lines)
                in_trainer = False
                # fall through to process this line normally
            else:
                # Skip blank lines and comments inside trainer block without
                # treating them as content that keeps the block non-empty.
                if not stripped or stripped.startswith("#"):
                    continue
                key = stripped.split(":", 1)[0].strip().split(" ")[0]
                if key in fw["trainer"]:
                    continue  # skip this trainer sub-key
                trainer_block_lines.append(l)
                continue
        if not in_trainer:
            # Top-level key check.
            if indent_len == 0 and ":" in l:
                key = stripped.split(":", 1)[0].strip().split(" ")[0]
                if key in fw["top"]:
                    continue  # skip
        out_lines.append(l)
    # Flush trailing trainer block if at EOF.
    if in_trainer and trainer_block_lines:
        out_lines.append("trainer:")
        out_lines.extend(trainer_block_lines)
    final_text = "\n".join(out_lines)
    # Add migration header comment if not already present.
    if "Migrated to inherit lineage" not in text:
        header = (
            "# @package _global_\n"
            f"# Migrated to inherit lineage/{lineage}.yaml (framework defaults).\n"
            "# Business params preserved verbatim; resolved config identical to pre-migration.\n"
        )
        # Replace the leading @package line if present.
        if final_text.lstrip().startswith("# @package _global_"):
            final_text = header + final_text.split("\n", 1)[1]
        else:
            final_text = header + final_text
    return final_text, is_root, True


def migrate_one(experiment: str, lineage: str) -> tuple[bool, str]:
    rel = f"config/experiment/{experiment}.yaml"
    baseline_path = BASELINE_DIR / rel
    if not baseline_path.exists():
        return False, f"baseline missing: {baseline_path}"
    text = baseline_path.read_text()
    new_text, is_root, changed = migrate_text(text, lineage)
    out_path = REPO / rel
    out_path.write_text(new_text)

    # Verify equivalence.
    r = subprocess.run(
        [sys.executable, "scripts/migrate/compare_resolved.py",
         "verify-one", experiment],
        cwd=str(REPO), capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
        return True, "OK (identical)"
    if "DIFF" in out:
        lines = out.split("\n")
        non_lineage = [l.strip().rstrip(":") for l in lines
                       if l.startswith("  ") and not l.startswith("    ")
                       and ":" in l and l.strip().rstrip(":") != "lineage"
                       and "keys differ" not in l]
        if not non_lineage:
            return True, "OK (only lineage key added)"
        return False, f"DIFFS {non_lineage[:6]}"
    return False, f"compose fail: {out[-400:]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage", choices=["emgformer", "fusion", "classic"])
    args = parser.parse_args()
    group = args.lineage
    experiments = list_experiments(group)
    print(f"Migrating {len(experiments)} {group} experiments to lineage/{group}.yaml")
    ok = fail = 0
    failed = []
    for exp in experiments:
        success, msg = migrate_one(exp, group)
        status = "✓" if success else "✗"
        print(f"  {status} {exp}: {msg[:200]}")
        if success:
            ok += 1
        else:
            fail += 1
            failed.append(exp)
    print(f"\nResult: {ok} OK, {fail} failed")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
