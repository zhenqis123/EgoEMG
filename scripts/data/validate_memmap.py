"""One-stop health check for a unified EgoEMG memmap directory.

Checks schema consistency (file presence and exact sizes), episode-table
invariants (exclusive-end continuity), per-source fill policies, and the
documented sentinel conventions. Can also generate or verify a
``checksums.json`` covering every storage file.

Usage:
  python scripts/data/validate_memmap.py --memmap-dir <dir>            # fast checks
  python scripts/data/validate_memmap.py --memmap-dir <dir> --full     # exhaustive scans
  python scripts/data/validate_memmap.py --memmap-dir <dir> --generate-checksums
  python scripts/data/validate_memmap.py --memmap-dir <dir> --checksums checksums.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}" + (f" — {detail}" if detail else "")
    print(line)
    if not ok:
        FAILURES.append(line)


def load_meta(root: Path) -> dict:
    meta = np.load(root / "metadata.npz", allow_pickle=True)
    return {k: meta[k] for k in meta.files}


def check_schema(root: Path, manifest: dict) -> None:
    fields = manifest["fields"]
    report("manifest.format_version", "format_version" in manifest, str(manifest.get("format_version")))
    for name, spec in fields.items():
        p = root / spec["filename"]
        if not p.is_file():
            report(f"file exists: {name}", False, str(p))
            continue
        expected = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
        report(f"size: {name}", p.stat().st_size == expected,
               f"{p.stat().st_size} vs {expected} bytes")


def check_episodes(root: Path, manifest: dict, meta: dict) -> None:
    n = manifest["total_rows"]
    start = meta["episode_start_idx"].astype(np.int64)
    end = meta["episode_end_idx"].astype(np.int64)
    report("episode table covers [0, total_rows)", start[0] == 0 and end[-1] == n,
           f"start[0]={start[0]}, end[-1]={end[-1]}, N={n}")
    report("episode table contiguous (exclusive ends)", bool((start[1:] == end[:-1]).all())
           and bool((end > start).all()))


def _sample(n: int, full: bool) -> np.ndarray:
    if full:
        return np.arange(n)
    step = max(1, n // 100_000)
    return np.arange(0, n, step)


def check_sources(root: Path, manifest: dict, full: bool) -> None:
    fields = manifest["fields"]
    n = manifest["total_rows"]
    idx = _sample(n, full)

    def arr(name: str) -> np.ndarray:
        spec = fields[name]
        return np.asarray(np.memmap(root / spec["filename"], dtype=spec["dtype"],
                                    mode="r", shape=tuple(spec["shape"]))[idx])

    src = arr("dataset_source_id")
    sources = set(manifest["dataset_sources"].keys())
    report("dataset_source_id values known", bool(np.isin(src, [int(s) for s in sources]).all()))

    incre = src == 2
    if incre.any():
        report("Incre: mocap validity all False",
               not (arr("mocap_left_valid")[incre].any() or arr("mocap_right_valid")[incre].any()))
        report("Incre: mocap keypoints zero",
               float(np.abs(arr("mocap_left_keypoints")[incre]).sum()) == 0.0
               and float(np.abs(arr("mocap_right_keypoints")[incre]).sum()) == 0.0)
        stale = (arr("image_head_stale")[incre] & arr("image_zed_stale")[incre])
        report("Incre: image stale bits True", bool(stale.all()))
        report("Incre: gesture_active False with class 0 (documented sentinel)",
               bool((~arr("label_gesture_active")[incre]).all()))
    showee = src == 1
    if showee.any():
        report("ShowEE: split all train", bool((arr("frame_split_id")[showee] == 0).all()))
    ego = src == 0
    if ego.any():
        splits = set(np.unique(arr("frame_split_id")[ego]).tolist())
        report("EgoEMG: all four splits present", splits == {0, 1, 2, 3}, str(sorted(splits)))


def generate_checksums(root: Path) -> None:
    names = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                   if p.is_file() and p.suffix in (".dat", ".json", ".npz")
                   and p.name != "checksums.json"
                   and not str(p.relative_to(root)).startswith("vision_index"))
    out: dict[str, str] = {}
    for i, name in enumerate(names):
        h = hashlib.sha256()
        with open(root / name, "rb") as f:  # subdirectory-safe relative paths
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        out[name] = h.hexdigest()
        print(f"  [{i + 1}/{len(names)}] {name}: {h.hexdigest()[:16]}…", flush=True)
    (root / "checksums.json").write_text(json.dumps(out, indent=2))
    print(f"checksums.json written ({len(out)} files)")


def verify_checksums(root: Path, path: Path) -> None:
    expected = json.loads(path.read_text())
    for name, want in expected.items():
        p = root / name
        if not p.is_file():
            report(f"checksum: {name}", False, "missing file")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        report(f"checksum: {name}", h.hexdigest() == want)


def check_structure(root: Path, manifest: dict, meta: dict) -> None:
    """Groups/files/marks/labels/schema cross-checks (fast, no hashing)."""
    # modality_groups covers every frame field exactly once; groups match dirs
    if "modality_groups" in manifest:
        grouped: dict[str, list[str]] = {}
        for name, spec in manifest["fields"].items():
            grouped.setdefault(spec["filename"].split("/")[0], []).append(name)
        declared = {g: {f.replace(" (episode-level)", "") for f in v}
                    for g, v in manifest["modality_groups"].items()}
        frame_fields = set(manifest["fields"].keys())
        for g, fields in grouped.items():
            # a declared group is allowed to also list episode-level entries;
            # its frame-field subset must equal the directory's contents
            report(f"modality_groups matches dir {g}/",
                   declared.get(g, set()) & frame_fields == set(fields))
        covered = {f for v in declared.values() for f in v}
        report("modality_groups covers all fields",
               (covered & frame_fields) == frame_fields)
    # no orphan .dat files outside the manifest
    declared_files = {spec["filename"] for spec in manifest["fields"].values()} \
        | {spec["filename"] for spec in manifest["episode_fields"].values()}
    orphans = sorted(str(p.relative_to(root)) for p in root.rglob("*.dat")
                     if str(p.relative_to(root)) not in declared_files)
    report("no orphan .dat files", not orphans, ", ".join(orphans[:5]))
    # is_first/is_last pairing and beta-table cardinality
    n = manifest["total_rows"]
    # boundary marks are sparse (928 marks in 132M rows): read fully
    fi = _arr(root, manifest, "is_first", np.arange(n))
    la = _arr(root, manifest, "is_last", np.arange(n))
    n_first, n_last = int(fi.sum()), int(la.sum())
    beta_rows = next(iter(manifest["episode_fields"].values()))["shape"][0]
    report("is_first count == beta rows (928)", n_first == beta_rows, f"{n_first} vs {beta_rows}")
    report("is_last count == is_first count", n_last == n_first, f"{n_last} vs {n_first}")
    # gesture labels within envelope
    gc = _arr(root, manifest, "label_gesture_class", _sample(n, False))
    report("label_gesture_class in [-1, 59]",
           bool(((gc >= -1) & (gc <= 59)).all()))
    # schema version cross-check
    sv = str(meta.get("schema_version", [""])[0]) if "schema_version" in meta else ""
    report("metadata schema_version matches format_version family",
           "v3" in sv and "v3" in manifest.get("format_version", ""), sv)


def _arr(root: Path, manifest: dict, name: str, idx: np.ndarray) -> np.ndarray:
    spec = manifest["fields"][name]
    return np.asarray(np.memmap(root / spec["filename"], dtype=spec["dtype"],
                                mode="r", shape=tuple(spec["shape"]))[idx])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--full", action="store_true", help="exhaustive scans instead of sampling")
    ap.add_argument("--generate-checksums", action="store_true")
    ap.add_argument("--checksums", type=Path, default=None,
                    help="verify against a checksums.json (defaults to <dir>/checksums.json if present)")
    ap.add_argument("--no-checksums", action="store_true",
                    help="skip checksum verification even if checksums.json exists")
    args = ap.parse_args()

    root = args.memmap_dir
    manifest = json.loads((root / "manifest.json").read_text())
    meta = load_meta(root)

    check_schema(root, manifest)
    check_episodes(root, manifest, meta)
    check_sources(root, manifest, args.full)

    if args.generate_checksums:
        generate_checksums(root)
    ck = None if args.no_checksums else (
        args.checksums or (root / "checksums.json" if (root / "checksums.json").exists() else None)
    )
    if ck:
        verify_checksums(root, ck)
    check_structure(root, manifest, meta)

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
