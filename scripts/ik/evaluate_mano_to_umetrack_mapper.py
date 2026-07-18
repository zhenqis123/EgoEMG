#!/usr/bin/env python3
"""Evaluate MANO-theta to UmeTrack-angle mapper against IK-fit labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emg2pose.kinematics import apply_to_hand_model, broadcast_hand_model_to
from emg2pose.realtime_local.mano_mapper import RuntimeManoToUmeTrackMapper
from emg2pose.UmeTrack.lib.common.hand_skinning import (
    _get_skinned_vertices,
    _hand_skinning_transform,
    _lbs,
    skin_landmarks,
)
from scripts.ik.train_mano_to_umetrack_mapper import (
    discover_segments,
    sample_batch,
)


def _umetrack_fk(
    hand_model,
    angles20: torch.Tensor,
    *,
    return_mesh: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    bsz = angles20.shape[0]
    hand_model_b = broadcast_hand_model_to(hand_model, (bsz,))
    hand_model_b = apply_to_hand_model(hand_model_b, lambda t: t.float().to(angles20.device))
    wrist_tf = torch.eye(4, device=angles20.device).unsqueeze(0).expand(bsz, -1, -1)
    landmarks = skin_landmarks(hand_model_b, angles20, wrist_transforms=wrist_tf)
    mesh = None
    if return_mesh:
        angles22 = torch.cat(
            [angles20, torch.zeros(bsz, 2, dtype=angles20.dtype, device=angles20.device)],
            dim=1,
        )
        skin_xfs = _hand_skinning_transform(
            hand_model_b.joint_rotation_axes.reshape(bsz, -1, 3),
            hand_model_b.joint_rest_positions.reshape(bsz, -1, 3),
            angles22,
            wrist_tf,
        )
        weights = hand_model_b.dense_bone_weights.reshape(bsz, -1, 17)
        rest_vertices = hand_model_b.mesh_vertices.reshape(bsz, -1, 3)
        skin_vertices = _get_skinned_vertices(rest_vertices, weights)
        mesh = _lbs(skin_xfs, skin_vertices)[..., :3]
    return landmarks, mesh


def _load_hand_model(device: torch.device):
    from emg2pose.kinematics import load_default_hand_model

    hand_model = load_default_hand_model()
    return apply_to_hand_model(hand_model, lambda t: t.float().to(device))


def _mano_teacher_landmarks(
    mano_layer,
    pose48: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    from scripts.ik.batch_ik_mesh import ALIGN_SCALE, ALIGN_TRANS, FLIP_MATRIX, MANO_IDX

    pose = pose48.clone()
    pose[:, :3] = 0.0
    beta = torch.zeros(pose.shape[0], 10, device=device)
    flip_t = torch.from_numpy(FLIP_MATRIX).float().to(device)
    trans = torch.tensor(ALIGN_TRANS.tolist(), dtype=torch.float32, device=device)
    with torch.no_grad():
        out = mano_layer(pose, beta)
        mano_j = out.joints * 1000.0
        aligned = ALIGN_SCALE * (mano_j @ flip_t.T) + trans.unsqueeze(0).unsqueeze(0)
    return aligned[:, MANO_IDX, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained_models/mano_to_umetrack_mapper.pt"))
    parser.add_argument("--data-root", type=Path, action="append", default=None)
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-max-angle", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mesh", action="store_true", help="Also compute all-vertex mesh distance.")
    parser.add_argument("--mano-geometry", action="store_true", help="Also compare fit/mapper to MANO teacher landmarks.")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = args.data_root
    if roots is None:
        from scripts.ik.train_mano_to_umetrack_mapper import default_roots

        roots = default_roots()
    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    segments = discover_segments(
        roots=roots,
        hands=hands,
        val_fraction=0.15,
        max_episode_frames=None,
    )
    if args.split != "all":
        segments = [s for s in segments if s.split == args.split]
    if not segments:
        raise SystemExit("No matching segments with paired MANO/angle fields.")

    print(f"segments={len(segments)} candidate_frames={sum(s.length for s in segments):,}")
    for segment in segments[:8]:
        print(
            f"  {segment.split:5s} {segment.hand:5s} {segment.root} "
            f"{segment.episode_name} [{segment.start}, {segment.end})"
        )
    if len(segments) > 8:
        print(f"  ... {len(segments) - 8} more")

    device = torch.device(args.device)
    mapper = RuntimeManoToUmeTrackMapper(args.checkpoint, device=device)
    hand_model = _load_hand_model(device)

    mano_layer = None
    if args.mano_geometry:
        from scripts.ik.batch_ik_mesh import MANO_ASSETS_ROOT
        from manotorch.manolayer import ManoLayer

        mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            mano_assets_root=str(MANO_ASSETS_ROOT),
            use_pca=False,
            flat_hand_mean=False,
        ).to(device)
        mano_layer.eval()

    rng = np.random.default_rng(args.seed)
    remaining = args.samples
    n_angles = 0
    angle_abs_sum = 0.0
    angle_max = 0.0
    lm_sum = 0.0
    lm_count = 0
    mesh_sum = 0.0
    mesh_count = 0
    fit_to_mano_sum = 0.0
    mapper_to_mano_sum = 0.0
    mano_count = 0

    while remaining > 0:
        cur = min(args.batch_size, remaining)
        pose45_np, fit_np = sample_batch(
            rng,
            segments,
            cur,
            angle_abs_limit=args.hidden_max_angle,
        )
        if pose45_np.shape[0] == 0:
            continue
        mapper_np = mapper.predict(pose45_np)
        fit = torch.from_numpy(fit_np).float().to(device)
        pred = torch.from_numpy(mapper_np).float().to(device)

        abs_err = (pred - fit).abs()
        angle_abs_sum += float(abs_err.sum().item())
        angle_max = max(angle_max, float(abs_err.max().item()))
        n_angles += int(abs_err.numel())

        fit_lm, fit_mesh = _umetrack_fk(hand_model, fit, return_mesh=args.mesh)
        pred_lm, pred_mesh = _umetrack_fk(hand_model, pred, return_mesh=args.mesh)
        lm_dist = torch.linalg.norm(pred_lm - fit_lm, dim=-1)
        lm_sum += float(lm_dist.sum().item())
        lm_count += int(lm_dist.numel())
        if args.mesh:
            assert fit_mesh is not None and pred_mesh is not None
            mesh_dist = torch.linalg.norm(pred_mesh - fit_mesh, dim=-1)
            mesh_sum += float(mesh_dist.sum().item())
            mesh_count += int(mesh_dist.numel())

        if mano_layer is not None:
            pose48 = torch.zeros(pose45_np.shape[0], 48, dtype=torch.float32, device=device)
            pose48[:, 3:48] = torch.from_numpy(pose45_np).float().to(device)
            target_lm = _mano_teacher_landmarks(mano_layer, pose48, device)
            from scripts.ik.batch_ik_mesh import UMETRACK_IDX

            fit_sel = fit_lm[:, UMETRACK_IDX, :]
            pred_sel = pred_lm[:, UMETRACK_IDX, :]
            fit_to_mano = torch.linalg.norm(fit_sel - target_lm, dim=-1)
            pred_to_mano = torch.linalg.norm(pred_sel - target_lm, dim=-1)
            fit_to_mano_sum += float(fit_to_mano.sum().item())
            mapper_to_mano_sum += float(pred_to_mano.sum().item())
            mano_count += int(fit_to_mano.numel())

        remaining -= cur

    metrics = {
        "samples": args.samples,
        "angle_mae_rad": angle_abs_sum / max(n_angles, 1),
        "angle_mae_deg": np.rad2deg(angle_abs_sum / max(n_angles, 1)).item(),
        "angle_max_abs_rad": angle_max,
        "angle_max_abs_deg": np.rad2deg(angle_max).item(),
        "umetrack_landmark_mean_mm": lm_sum / max(lm_count, 1),
    }
    if args.mesh:
        metrics["umetrack_mesh_mean_mm"] = mesh_sum / max(mesh_count, 1)
    if mano_layer is not None:
        metrics["fit_to_mano_landmark_mean_mm"] = fit_to_mano_sum / max(mano_count, 1)
        metrics["mapper_to_mano_landmark_mean_mm"] = mapper_to_mano_sum / max(mano_count, 1)
        metrics["extra_mano_landmark_error_mm"] = (
            metrics["mapper_to_mano_landmark_mean_mm"]
            - metrics["fit_to_mano_landmark_mean_mm"]
        )

    print(json.dumps(metrics, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
