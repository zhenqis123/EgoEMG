#!/usr/bin/env python3
"""Conservatively prune completed training logs.

The script protects checkpoint/config paths referenced by repository artifacts,
keeps every minimum-val_mae checkpoint in a run, and only removes duplicate
checkpoints plus an explicit allowlist of known-invalid diagnostic runs.
Run without ``--apply`` first to inspect the dry-run plan.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


VAL_RE = re.compile(r"val_mae(?:=|:|-)(?:val_mae=)?([0-9]+(?:\.[0-9]+)?)")
LOG_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])logs/[^\s\"'`<>]+")


def referenced_paths(repo: Path, logs_root: Path) -> set[Path]:
    protected: set[Path] = set()
    scan_roots = [repo / name for name in ("test_results", "paper", "docs", "scripts", "config")]
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in {".json", ".md", ".txt", ".sh", ".yaml", ".yml"}
                or path.stat().st_size > 20 * 1024 * 1024
            ):
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for match in LOG_TOKEN_RE.finditer(text):
                token = match.group(0).rstrip(".,;:)]}")
                relative = token.split("logs/", 1)[1].split("\x00", 1)[0]
                try:
                    candidate = (logs_root / relative).resolve()
                except (OSError, ValueError):
                    continue
                if candidate.exists():
                    protected.add(candidate)
    return protected


def duplicate_candidates(logs_root: Path, protected: set[Path]) -> list[Path]:
    candidates: list[Path] = []
    for checkpoint_dir in logs_root.rglob("checkpoints"):
        if not checkpoint_dir.is_dir():
            continue
        files = [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_file() and path.suffix in {".ckpt", ".pt", ".pth"}
        ]
        scored: list[tuple[float, Path]] = []
        for path in files:
            match = VAL_RE.search(path.name)
            if match:
                scored.append((float(match.group(1)), path))
        if not scored:
            continue
        best = min(score for score, _ in scored)
        keep = {path.resolve() for score, path in scored if score == best}
        keep.update(path for path in (p.resolve() for p in files) if path in protected)
        for path in files:
            resolved = path.resolve()
            if resolved in keep:
                continue
            # Unscored files are normally last/resume snapshots. They are safe
            # to remove only when a scored checkpoint exists and they are not
            # referenced by an artifact.
            if VAL_RE.search(path.name) or path.name.startswith(("last", "resume")):
                candidates.append(path)
    return candidates


def explicit_invalid_runs(logs_root: Path) -> list[Path]:
    relative = [
        "20260728/sensingdynamics_egoemg_smoke",
        "20260728/wilor_s_corrected_crop_1batch_probe",
        "20260728/wilor_s_exact_vision_1batch_probe",
        "20260728/fusion_5vision_s_simple_unfrozen_augbest_30e/wilor_s_invalid_frozen_bug",
        "20260728/fusion_5vision_s_simple_unfrozen_augbest_30e/wilor_s_invalid_unused_mano",
        "20260729_vitb_lastblock",
        "20260729_vitb_lastblock_v2lr1e7",
    ]
    return [logs_root / item for item in relative if (logs_root / item).exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    logs_root = (args.logs_root or repo / "logs").resolve()
    protected = referenced_paths(repo, logs_root)
    duplicates = duplicate_candidates(logs_root, protected)
    invalid = explicit_invalid_runs(logs_root)

    duplicate_bytes = sum(path.stat().st_size for path in duplicates)
    invalid_bytes = sum(
        sum(item.stat().st_size for item in run.rglob("*") if item.is_file())
        for run in invalid
    )
    print(f"Protected referenced paths: {len(protected)}")
    print(f"Duplicate checkpoint files: {len(duplicates)} ({duplicate_bytes / 1024**3:.2f} GB)")
    print(f"Explicit invalid/diagnostic runs: {len(invalid)} ({invalid_bytes / 1024**3:.2f} GB)")
    for run in invalid:
        print(f"  INVALID {'DELETE' if args.apply else 'WOULD DELETE'} {run}")
    for path in sorted(duplicates):
        print(f"  CKPT {'DELETE' if args.apply else 'WOULD DELETE'} {path}")

    if not args.apply:
        print("Dry run only; rerun with --apply to execute this allowlisted cleanup.")
        return

    for path in duplicates:
        if path.exists() and path.resolve() not in protected:
            path.unlink()
    for run in invalid:
        if run.exists():
            shutil.rmtree(run)
    print("Cleanup completed.")


if __name__ == "__main__":
    main()
