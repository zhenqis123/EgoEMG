#!/usr/bin/env python
"""Diagnose modality use and gradient flow in frozen-vision fusion models.

The script never calls ``optimizer.step``. It evaluates controlled input and
feature interventions, then runs a small number of backward-only batches to
measure gradient RMS by component.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egoemg.datamodule import make_data_module


REPO = Path(__file__).resolve().parents[2]
MODES = (
    "normal",
    "zero_emg",
    "shuffle_emg",
    "no_cross_attention",
    "shuffle_visual_tokens",
    "vision_only",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        default=(
            "fusion/"
            "fusion_rn50_m_egoemg_only_noaug_wl12000_frozen_crossattn_100e"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-batches", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(args: argparse.Namespace) -> tuple[Any, Any]:
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "config")
    ):
        cfg = compose(
            config_name="base",
            overrides=[
                f"experiment={args.experiment}",
                f"batch_size={args.batch_size}",
                f"num_workers={args.num_workers}",
            ],
        )
    datamodule = make_data_module(cfg)
    datamodule.setup("fit")

    model = instantiate(cfg.module)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model_state = {
        key[len("model.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)
    return model, datamodule


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _no_cross_attention(model: Any, emg_features: torch.Tensor) -> torch.Tensor:
    fusion = model.early_fusion
    tokens = emg_features.transpose(1, 2)
    tokens = tokens + fusion.ffn(fusion.ffn_norm(tokens))
    return tokens.transpose(1, 2)


def _decode_center(
    model: Any,
    features: torch.Tensor,
    vision_pose: torch.Tensor,
) -> torch.Tensor:
    decoded = model.decoder(features)
    center = decoded.shape[-1] // 2
    delta = model.head(decoded[..., center : center + 1])
    return vision_pose.unsqueeze(-1) + delta


def _update_metric(
    store: dict[str, dict[str, list[float]]],
    mode: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    hands: torch.Tensor,
) -> None:
    error = (prediction - target).abs()
    valid = valid.reshape(valid.shape[0], -1).any(dim=1)
    for hand_index, hand_name in ((0, "left"), (1, "right")):
        selected = valid & (hands == hand_index)
        if not selected.any():
            continue
        values = error[selected]
        store[mode][hand_name][0] += float(values.sum())
        store[mode][hand_name][1] += int(values.numel())


def _evaluate(
    model: Any,
    dataloader: Any,
    device: torch.device,
    max_batches: int | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    store = {
        mode: {"left": [0.0, 0], "right": [0.0, 0]} for mode in MODES
    }
    activation_sums = defaultdict(float)
    activation_count = 0
    model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _to_device(batch, device)
            emg = batch["emg"]
            valid = batch["vision_valid_mask"]
            if valid.ndim > 1:
                valid = valid.any(dim=1)

            layer3, layer4 = model._extract_resnet_multiscale(
                batch["vision_img"]
            )
            vision_features = layer4.mean(dim=(-2, -1))
            vision_features = vision_features * valid[:, None].to(
                vision_features.dtype
            )
            vision_pose = model.head_vision(vision_features)
            emg_features = model.featurizer(emg)

            permutation = torch.roll(
                torch.arange(emg.shape[0], device=device), shifts=1
            )
            normal_features = model.early_fusion(
                emg_features, layer3, layer4, valid
            )
            predictions = {
                "normal": _decode_center(
                    model, normal_features, vision_pose
                ),
                "zero_emg": _decode_center(
                    model,
                    model.early_fusion(
                        model.featurizer(torch.zeros_like(emg)),
                        layer3,
                        layer4,
                        valid,
                    ),
                    vision_pose,
                ),
                "shuffle_emg": _decode_center(
                    model,
                    model.early_fusion(
                        emg_features[permutation], layer3, layer4, valid
                    ),
                    vision_pose,
                ),
                "no_cross_attention": _decode_center(
                    model,
                    _no_cross_attention(model, emg_features),
                    vision_pose,
                ),
                "shuffle_visual_tokens": _decode_center(
                    model,
                    model.early_fusion(
                        emg_features,
                        layer3[permutation],
                        layer4[permutation],
                        valid,
                    ),
                    vision_pose,
                ),
                "vision_only": vision_pose.unsqueeze(-1),
            }

            for mode, prediction in predictions.items():
                _update_metric(
                    store,
                    mode,
                    prediction,
                    batch["joint_angles"],
                    batch["label_valid_mask"],
                    batch["target_hand_index"],
                )

            normal = predictions["normal"]
            activation_sums["residual_sq"] += float(
                (normal - predictions["vision_only"]).square().sum()
            )
            activation_sums["zero_emg_change_sq"] += float(
                (normal - predictions["zero_emg"]).square().sum()
            )
            activation_sums["shuffle_emg_change_sq"] += float(
                (normal - predictions["shuffle_emg"]).square().sum()
            )
            activation_sums["no_cross_change_sq"] += float(
                (normal - predictions["no_cross_attention"]).square().sum()
            )
            activation_sums["shuffle_visual_change_sq"] += float(
                (normal - predictions["shuffle_visual_tokens"]).square().sum()
            )
            activation_sums["feature_injection_sq"] += float(
                (normal_features - emg_features).square().sum()
            )
            activation_sums["emg_feature_sq"] += float(
                emg_features.square().sum()
            )
            activation_count += normal.numel()

            if (batch_index + 1) % 20 == 0:
                print(f"evaluated {batch_index + 1}/{len(dataloader)} batches")

    results: dict[str, Any] = {}
    for mode in MODES:
        left_sum, left_count = store[mode]["left"]
        right_sum, right_count = store[mode]["right"]
        total_sum = left_sum + right_sum
        total_count = left_count + right_count
        results[mode] = {
            "combined_mae": total_sum / total_count,
            "left_mae": left_sum / left_count,
            "right_mae": right_sum / right_count,
            "elements": total_count,
        }

    activations = {
        "residual_rms": math.sqrt(
            activation_sums["residual_sq"] / activation_count
        ),
        "zero_emg_prediction_change_rms": math.sqrt(
            activation_sums["zero_emg_change_sq"] / activation_count
        ),
        "shuffle_emg_prediction_change_rms": math.sqrt(
            activation_sums["shuffle_emg_change_sq"] / activation_count
        ),
        "no_cross_prediction_change_rms": math.sqrt(
            activation_sums["no_cross_change_sq"] / activation_count
        ),
        "shuffle_visual_prediction_change_rms": math.sqrt(
            activation_sums["shuffle_visual_change_sq"] / activation_count
        ),
        "feature_injection_ratio": math.sqrt(
            activation_sums["feature_injection_sq"]
            / activation_sums["emg_feature_sq"]
        ),
    }
    return results, activations


def _gradient_family(name: str) -> str:
    prefixes = (
        "early_fusion.layer3_proj",
        "early_fusion.layer4_proj",
        "early_fusion.cross_attention",
        "early_fusion.ffn",
        "featurizer",
        "decoder",
        "head",
    )
    return next(
        (prefix for prefix in prefixes if name.startswith(prefix)), "other"
    )


def _gradient_diagnostics(
    model: Any,
    dataloader: Any,
    device: torch.device,
    num_batches: int,
) -> dict[str, Any]:
    accumulated = defaultdict(
        lambda: {"grad_sq": 0.0, "weight_sq": 0.0, "elements": 0, "batches": 0}
    )
    model.eval()
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= num_batches:
            break
        batch = _to_device(batch, device)
        model.zero_grad(set_to_none=True)
        prediction, target, mask = model(batch)
        valid = mask.to(prediction.dtype)
        if valid.ndim == 2:
            valid = valid[:, None, :]
        loss = (
            (prediction - target).abs() * valid
        ).sum() / (valid.sum().clamp_min(1) * prediction.shape[1])
        loss.backward()

        per_batch = defaultdict(lambda: [0.0, 0.0, 0])
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            family = _gradient_family(name)
            per_batch[family][0] += float(parameter.grad.float().square().sum())
            per_batch[family][1] += float(parameter.float().square().sum())
            per_batch[family][2] += parameter.numel()
        for family, (grad_sq, weight_sq, elements) in per_batch.items():
            accumulated[family]["grad_sq"] += grad_sq
            accumulated[family]["weight_sq"] += weight_sq
            accumulated[family]["elements"] += elements
            accumulated[family]["batches"] += 1

    output = {}
    for family, values in accumulated.items():
        elements = values["elements"]
        output[family] = {
            "gradient_rms": math.sqrt(values["grad_sq"] / elements),
            "weight_rms": math.sqrt(values["weight_sq"] / elements),
            "gradient_over_weight": math.sqrt(
                values["grad_sq"] / max(values["weight_sq"], 1e-30)
            ),
            "batches": values["batches"],
        }
    model.zero_grad(set_to_none=True)
    return output


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    model, datamodule = _load(args)
    model.to(device)
    dataloader = datamodule.val_dataloader()

    interventions, activations = _evaluate(
        model, dataloader, device, args.max_eval_batches
    )
    gradients = _gradient_diagnostics(
        model, dataloader, device, args.gradient_batches
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "interventions": interventions,
        "activations": activations,
        "gradients": gradients,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
