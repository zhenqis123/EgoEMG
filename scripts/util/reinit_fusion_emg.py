"""Re-initialize dead EMG branch components in a fusion checkpoint.

Reloads featurizer + decoder from a pretrained EMG checkpoint, re-initializes
fusion_proj + head + gate_proj + residual_scale, and keeps vision branch
(vision_proj, head_vision) unchanged.

Usage:
    python scripts/util/reinit_fusion_emg.py \\
        --input logs/fusion/.../version_13/checkpoints/last.ckpt \\
        --emg-ckpt logs/2026-04-29/.../egoemg-epoch=087-val_mae=0.2629.ckpt \\
        --output logs/fusion/.../version_13/checkpoints/reinit.ckpt
"""

import argparse
import torch
from torch import nn


def reload_featurizer_decoder(sd: dict, emg_ckpt_path: str) -> dict:
    """Copy featurizer + decoder weights from a pretrained EMG checkpoint."""
    emg = torch.load(emg_ckpt_path, map_location="cpu", weights_only=False)
    emg_sd = emg.get("state_dict", emg)

    matched = 0
    for key, value in emg_sd.items():
        # Strip "model." prefix if present
        stripped = key[6:] if key.startswith("model.") else key
        if not stripped.startswith(("featurizer.", "decoder.")):
            continue
        target_key = f"model.{stripped}"
        if target_key in sd and sd[target_key].shape == value.shape:
            sd[target_key] = value.clone()
            matched += 1

    print(f"  Reloaded {matched} featurizer/decoder keys from {emg_ckpt_path}")
    return sd


def reinit_emg_branch(sd: dict, emg_ckpt_path: str) -> dict:
    """Re-initialize fusion_proj, head, gate_proj, and residual_scale.

    Also reloads featurizer + decoder from a pretrained EMG checkpoint.
    Keeps vision_proj, head_vision unchanged.
    """
    sd = dict(sd)  # shallow copy

    # ── Reload featurizer + decoder from pretrained EMG checkpoint ─────────
    sd = reload_featurizer_decoder(sd, emg_ckpt_path)

    # ── fusion_proj: Conv1d layers — default init ──────────────────────────
    for key in ("model.fusion_proj.0.weight", "model.fusion_proj.0.bias",
                "model.fusion_proj.3.weight", "model.fusion_proj.3.bias"):
        if key in sd:
            t = sd[key]
            if "weight" in key and t.ndim >= 2:
                nn.init.kaiming_normal_(t, mode="fan_out", nonlinearity="relu")
            elif "bias" in key:
                nn.init.zeros_(t)
            print(f"  Re-init: {key}")

    # ── head.net.0: first Linear (256→512) — default init ────────────────
    for suffix in ("weight", "bias"):
        key = f"model.head.net.0.{suffix}"
        if key in sd:
            t = sd[key]
            if suffix == "weight":
                nn.init.kaiming_normal_(t, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.zeros_(t)
            print(f"  Re-init: {key}")

    # ── head.net.3: last Linear (512→22) — near-zero init ────────────────
    for suffix in ("weight", "bias"):
        key = f"model.head.net.3.{suffix}"
        if key in sd:
            t = sd[key]
            if suffix == "weight":
                nn.init.normal_(t, mean=0.0, std=1e-5)
            else:
                nn.init.zeros_(t)
            print(f"  Re-init: {key}")

    # ── gate_proj.0: first Linear (256→64) — default init ────────────────
    for suffix in ("weight", "bias"):
        key = f"model.gate_proj.0.{suffix}"
        if key in sd:
            t = sd[key]
            if suffix == "weight":
                nn.init.kaiming_normal_(t, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.zeros_(t)
            print(f"  Re-init: {key}")

    # ── gate_proj.2: last Linear (64→22) — small init, bias=0 ────────────
    for suffix in ("weight", "bias"):
        key = f"model.gate_proj.2.{suffix}"
        if key in sd:
            t = sd[key]
            if suffix == "weight":
                nn.init.normal_(t, mean=0.0, std=0.01)
            else:
                nn.init.zeros_(t)
            print(f"  Re-init: {key}")

    # ── residual_scale ───────────────────────────────────────────────────
    key = "model.residual_scale"
    if key in sd:
        sd[key] = torch.tensor(1e-3)
        print(f"  Re-init: {key} → 1e-3")

    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Fusion checkpoint with dead EMG branch")
    parser.add_argument("--emg-ckpt", type=str, required=True,
                        help="Pretrained EMG checkpoint for featurizer + decoder reload")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    original_sd = ckpt.get("state_dict", ckpt)

    print("Re-initializing EMG branch components:")
    new_sd = reinit_emg_branch(original_sd, args.emg_ckpt)

    # Write back into checkpoint wrapper
    if "state_dict" in ckpt:
        ckpt["state_dict"] = new_sd
    else:
        ckpt = new_sd

    # Clear optimizer states so training starts fresh
    for key in list(ckpt.keys()):
        if key in ("optimizer_states", "lr_schedulers", "optimizer_state_dict",
                    "lr_scheduler_state_dict"):
            del ckpt[key]
            print(f"  Removed: {key}")

    torch.save(ckpt, args.output)
    print(f"\nSaved re-initialized checkpoint to {args.output}")


if __name__ == "__main__":
    main()
