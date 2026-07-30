#!/usr/bin/env python3
"""Extract metadata from emg2qwerty zarr to memmap directory."""
from __future__ import annotations

import argparse
import json
import numpy as np
import zarr
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-dir", type=Path, required=True)
    parser.add_argument("--memmap-dir", type=Path, required=True)
    args = parser.parse_args()

    zarr_root = zarr.open(str(args.zarr_dir), mode="r")
    memmap_dir = Path(args.memmap_dir)
    memmap_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}

    # Session arrays
    print("Extracting sessions...")
    sessions = zarr_root["sessions"]
    for key in sessions.keys():
        arr = sessions[key]
        metadata[f"session_{key}"] = arr[:]
        print(f"  session_{key}: shape={arr.shape}, dtype={arr.dtype}")

    # Users
    print("Extracting users...")
    users = zarr_root["users"]
    for key in users.keys():
        arr = users[key]
        metadata[f"users_{key}"] = arr[:]
        print(f"  users_{key}: shape={arr.shape}")

    # Conditions
    print("Extracting conditions...")
    conditions = zarr_root["conditions"]
    for key in conditions.keys():
        arr = conditions[key]
        metadata[f"conditions_{key}"] = arr[:]
        print(f"  conditions_{key}: shape={arr.shape}")

    # Keystrokes (event data)
    print("Extracting keystrokes...")
    keystrokes = zarr_root["keystrokes"]
    for key in keystrokes.keys():
        arr = keystrokes[key]
        metadata[f"keystrokes_{key}"] = arr[:]
        print(f"  keystrokes_{key}: shape={arr.shape}, dtype={arr.dtype}")

    # Prompts (event data)
    print("Extracting prompts...")
    prompts = zarr_root["prompts"]
    for key in prompts.keys():
        arr = prompts[key]
        metadata[f"prompts_{key}"] = arr[:]
        print(f"  prompts_{key}: shape={arr.shape}, dtype={arr.dtype}")

    # Save to npz
    npz_path = memmap_dir / "metadata.npz"
    np.savez(npz_path, **metadata)
    print(f"\nSaved metadata.npz: {npz_path.stat().st_size / 1e6:.1f} MB")

    # Update manifest.json with session info
    manifest_path = memmap_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"total_rows": 0, "fields": {}}

    # Add session count
    manifest["num_sessions"] = len(metadata["session_start_idx"])
    manifest["num_users"] = len(metadata["users_user"])
    manifest["num_keystrokes"] = len(metadata["keystrokes_start"])
    manifest["num_prompts"] = len(metadata["prompts_start"])

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated manifest.json")

    # Print summary
    print("\n=== Summary ===")
    print(f"Total rows: {manifest['total_rows']:,}")
    print(f"Sessions: {manifest['num_sessions']}")
    print(f"Users: {manifest['num_users']}")
    print(f"Keystroke events: {manifest['num_keystrokes']:,}")
    print(f"Prompt events: {manifest['num_prompts']:,}")


if __name__ == "__main__":
    main()