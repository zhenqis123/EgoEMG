"""Fill a ShowEE memmap shard with markers2mano pose and beta labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from emg2pose.realtime_local.mano_mapper import RuntimeManoToUmeTrackMapper
from scripts.mano.infer_mano_for_egoemg import (
    _save_glb,
    build_viz_records,
    infer_hand,
    load_mano_layer,
    load_model,
    resolve_device,
)


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8").rstrip("\x00")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in values
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_field(
    root: Path, manifest: dict[str, Any], name: str, mode: str
) -> np.memmap:
    info = manifest["fields"][name]
    return np.memmap(
        root / info["filename"],
        mode=mode,
        dtype=info["dtype"],
        shape=tuple(info["shape"]),
    )


def _open_episode_field(
    root: Path, manifest: dict[str, Any], name: str, mode: str
) -> np.memmap:
    info = manifest["episode_fields"][name]
    return np.memmap(
        root / info["filename"],
        mode=mode,
        dtype=info["dtype"],
        shape=tuple(info["shape"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--viz-frames-per-hand", type=int, default=4)
    parser.add_argument(
        "--angle-mapper-checkpoint",
        type=Path,
        default=Path("pretrained_models/mano_to_umetrack_mapper.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.memmap_root.resolve()
    checkpoint = args.checkpoint.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = np.load(root / "metadata.npz", allow_pickle=False)
    source_root = Path(manifest["source_root"])
    source_paths = _decode(metadata["episode_source_parquet"])
    starts = metadata["episode_start_idx"]
    ends = metadata["episode_end_idx"]

    device = resolve_device(args.device)
    model = load_model(checkpoint, device)
    mano_layer = load_mano_layer(device)
    faces = mano_layer.th_faces.detach().cpu().numpy()

    pose = {
        hand: _open_field(root, manifest, f"generated_mano_{hand}_pose", "r+")
        for hand in ("left", "right")
    }
    beta = {
        hand: _open_episode_field(
            root, manifest, f"generated_mano_{hand}_beta", "r+"
        )
        for hand in ("left", "right")
    }
    label_valid = _open_field(root, manifest, "generated_label_valid", "r+")
    viz_dir = root / "mano_viz"
    viz_dir.mkdir(exist_ok=True)
    episode_reports: list[dict[str, Any]] = []

    for episode_idx, relative_path in enumerate(source_paths):
        task_dir = source_root / relative_path
        with h5py.File(task_dir / "luster_mocap" / "mocap.h5", "r") as handle:
            markers = {
                hand: (
                    handle[f"{hand}_hand/markers"][:].astype(np.float32) / 1000.0
                )
                for hand in ("left", "right")
            }
            # markers2mano needs the complete ordered 21-marker set.  Some raw
            # captures contain NaN for an entire tracking interval; never feed
            # partially missing frames to the network or rigid-alignment SVD.
            valid = {}
            for hand in ("left", "right"):
                complete = np.isfinite(markers[hand]).all(axis=(1, 2))
                valid[hand] = np.repeat(complete[:, None], 21, axis=1)

        start = int(starts[episode_idx])
        end = int(ends[episode_idx])
        target_count = end - start
        report: dict[str, Any] = {
            "episode": relative_path,
            "source_mocap_rows": len(markers["left"]),
            "target_emg_rows": target_count,
            "hands": {},
        }
        for hand_idx, hand in enumerate(("left", "right")):
            result = infer_hand(
                model=model,
                mano_layer=mano_layer,
                keypoints=markers[hand],
                valid=valid[hand],
                stride=args.stride,
                batch_size=args.batch_size,
                device=device,
                viz_frames=args.viz_frames_per_hand,
                left_hand_mode="flip_local_z" if hand == "left" else "none",
            )
            native_pose = result["pose_full"]
            source_idx = np.rint(
                np.linspace(0, len(native_pose) - 1, target_count)
            ).astype(np.int64)
            pose[hand][start:end] = native_pose[source_idx]
            beta[hand][episode_idx] = result["beta_mean"]
            frame_valid = np.any(valid[hand], axis=1)
            label_valid[start:end, hand_idx] = frame_valid[source_idx]
            records = build_viz_records(
                episode_stem=f"episode_{episode_idx:06d}",
                hand_side=hand,
                hand_result=result,
                mano_layer=mano_layer,
                device=device,
                faces=faces,
                viz_dir=viz_dir,
                left_hand_mode="flip_local_z",
            )
            for record in records:
                _save_glb(record)
            report["hands"][hand] = {
                "valid_ratio": float(frame_valid.mean()),
                "aligned_error_mm": float(result["aligned_error_mm"]),
                "beta": result["beta_mean"].tolist(),
                "visualizations": len(records),
            }
            print(
                f"[{episode_idx + 1}/{len(source_paths)}] {hand}: "
                f"error={result['aligned_error_mm']:.3f} mm"
            )
        episode_reports.append(report)

    for memmap in (*pose.values(), *beta.values(), label_valid):
        memmap.flush()

    angle_mapper_path = args.angle_mapper_checkpoint.resolve()
    angle_mapper = RuntimeManoToUmeTrackMapper(angle_mapper_path, device=device)
    angle_fields: dict[str, np.memmap] = {}
    for hand_idx, hand in enumerate(("left", "right")):
        name = f"generated_joint_angles_{hand}"
        filename = f"{name}.dat"
        manifest["fields"][name] = {
            "filename": filename,
            "dtype": "float32",
            "shape": [int(manifest["total_rows"]), 20],
        }
        angle_fields[hand] = np.memmap(
            root / filename,
            mode="w+",
            dtype=np.float32,
            shape=(int(manifest["total_rows"]), 20),
        )
        for start in range(0, int(manifest["total_rows"]), 8192):
            end = min(start + 8192, int(manifest["total_rows"]))
            predicted = angle_mapper.predict(np.asarray(pose[hand][start:end]))
            predicted[~np.asarray(label_valid[start:end, hand_idx])] = 0.0
            angle_fields[hand][start:end] = predicted
        angle_fields[hand].flush()
    manifest["generated_joint_angles_semantics"] = [
        "thumb_cmc_fe",
        "thumb_cmc_aa",
        "thumb_mcp_fe",
        "thumb_ip_fe",
        "index_mcp_aa",
        "index_mcp_fe",
        "index_pip_fe",
        "index_dip_fe",
        "middle_mcp_aa",
        "middle_mcp_fe",
        "middle_pip_fe",
        "middle_dip_fe",
        "ring_mcp_aa",
        "ring_mcp_fe",
        "ring_pip_fe",
        "ring_dip_fe",
        "pinky_mcp_aa",
        "pinky_mcp_fe",
        "pinky_pip_fe",
        "pinky_dip_fe",
    ]
    manifest["mano_to_joint_angles"] = {
        "checkpoint": str(angle_mapper_path),
        "sha256": _sha256(angle_mapper_path),
    }
    manifest["markers2mano"] = {
        "checkpoint": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "left_hand_strategy": "flip_local_z",
        "pose_semantics": "canonical_mano_right_for_both_hands",
        "inference_axis": "native_120hz_mocap_then_nearest_resample_to_emg",
        "stride": args.stride,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {"markers2mano": manifest["markers2mano"], "episodes": episode_reports}
    (root / "markers2mano_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Done: {root}")


if __name__ == "__main__":
    main()
