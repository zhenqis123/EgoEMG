#!/usr/bin/env python3
"""Report unreleased/developer-local references in active experiment configs.

This is intentionally a reporting tool, not a migrator: code pre-release keeps
research recipes for inspection but must not present them as portable commands.
Exit status is zero when the scan itself succeeds; use ``--strict`` in a future
release gate once every public recipe has been made portable.
"""
from __future__ import annotations

import argparse
from pathlib import Path


PATTERNS = ("../WiLoR", "../manotorch", "/logs/", "/test_results/", "/data/experiment_inputs/")


def find_nonportable_references(config_root: Path) -> list[tuple[Path, int, str]]:
    """Return active-config lines that need private assets or local directories."""
    findings: list[tuple[Path, int, str]] = []
    for path in sorted((config_root / "experiment").rglob("*.yaml")):
        if "_archive" in path.parts:
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if any(pattern in line for pattern in PATTERNS):
                findings.append((path, line_number, line.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--strict", action="store_true", help="Fail if references are found.")
    args = parser.parse_args()
    findings = find_nonportable_references(args.config_root)
    if not findings:
        print("No developer-local references found in active experiment configs.")
        return 0
    print("Research-only config references (not portable release recipes):")
    for path, line_number, line in findings:
        print(f"  {path}:{line_number}: {line}")
    print(f"\nFound {len(findings)} reference(s).")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
