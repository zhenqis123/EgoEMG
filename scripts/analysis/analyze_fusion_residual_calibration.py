#!/usr/bin/env python
"""Measure fusion/vision complementarity and residual-scale calibration.

Inputs are aligned NPZ files emitted by ``unified_center_eval.py
--predictions-dir``.  The leave-one-episode-out estimate selects one global
residual scale on all other episodes and applies it to the held-out episode,
avoiding evaluation of a scale on the same samples used to choose it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--fusion-name", required=True)
    parser.add_argument("--vision-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-steps", type=int, default=401)
    return parser.parse_args()


def load_aligned(
    directory: Path, fusion_name: str, vision_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = []
    vision_predictions = []
    targets = []
    episodes = []
    hands = []
    for hand_index, hand in enumerate(("left", "right")):
        fusion = np.load(directory / f"{fusion_name}_{hand}.npz")
        vision = np.load(directory / f"{vision_name}_{hand}.npz")
        for key in ("episode_indices", "centers"):
            if not np.array_equal(fusion[key], vision[key]):
                raise ValueError(f"unaligned {key} for {hand}")
        if not np.allclose(fusion["targets"], vision["targets"]):
            raise ValueError(f"unaligned targets for {hand}")
        n = len(fusion["predictions"])
        predictions.append(fusion["predictions"])
        vision_predictions.append(vision["predictions"])
        targets.append(fusion["targets"])
        episodes.append(fusion["episode_indices"])
        hands.append(np.full(n, hand_index, dtype=np.int64))
    return tuple(
        np.concatenate(values, axis=0)
        for values in (predictions, vision_predictions, targets, episodes, hands)
    )


def mae_at_alphas(
    vision: np.ndarray,
    residual: np.ndarray,
    target: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [np.abs(vision + alpha * residual - target).mean() for alpha in alphas]
    )


def main() -> None:
    args = parse_args()
    fusion, vision, target, episodes, hands = load_aligned(
        args.predictions_dir, args.fusion_name, args.vision_name
    )
    residual = fusion - vision
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    curve = mae_at_alphas(vision, residual, target, alphas)
    best_index = int(curve.argmin())

    heldout_errors = []
    heldout_alphas = []
    for episode in np.unique(episodes):
        train = episodes != episode
        test = ~train
        train_curve = mae_at_alphas(
            vision[train], residual[train], target[train], alphas
        )
        alpha = float(alphas[int(train_curve.argmin())])
        heldout_alphas.append(alpha)
        heldout_errors.append(
            np.abs(vision[test] + alpha * residual[test] - target[test]).reshape(-1)
        )
    loeo_mae = float(np.concatenate(heldout_errors).mean())

    fusion_sample_mae = np.abs(fusion - target).mean(axis=1)
    vision_sample_mae = np.abs(vision - target).mean(axis=1)
    sample_oracle = float(np.minimum(fusion_sample_mae, vision_sample_mae).mean())
    element_oracle = float(
        np.minimum(np.abs(fusion - target), np.abs(vision - target)).mean()
    )

    result = {
        "n_samples": int(len(target)),
        "vision_mae": float(np.abs(vision - target).mean()),
        "fusion_alpha_1_mae": float(np.abs(fusion - target).mean()),
        "best_in_sample_alpha": float(alphas[best_index]),
        "best_in_sample_mae": float(curve[best_index]),
        "leave_one_episode_out_mae": loeo_mae,
        "leave_one_episode_out_alpha_mean": float(np.mean(heldout_alphas)),
        "leave_one_episode_out_alpha_std": float(np.std(heldout_alphas)),
        "leave_one_episode_out_alpha_min": float(np.min(heldout_alphas)),
        "leave_one_episode_out_alpha_max": float(np.max(heldout_alphas)),
        "fusion_wins_sample_fraction": float(
            np.mean(fusion_sample_mae < vision_sample_mae)
        ),
        "sample_oracle_mae": sample_oracle,
        "element_oracle_mae": element_oracle,
        "by_hand": {},
    }
    for hand_index, hand_name in enumerate(("left", "right")):
        selected = hands == hand_index
        hand_curve = mae_at_alphas(
            vision[selected], residual[selected], target[selected], alphas
        )
        index = int(hand_curve.argmin())
        result["by_hand"][hand_name] = {
            "best_alpha": float(alphas[index]),
            "best_mae": float(hand_curve[index]),
            "alpha_1_mae": float(np.abs(fusion[selected] - target[selected]).mean()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
