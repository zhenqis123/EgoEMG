from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from egoemg.datasets.layout_utils import place_sparse_channels
from egoemg.models.decoders.transformer import TransformerDecoder
from egoemg.models.featurizers.tds import Conv1dBlock, TdsNetwork, TdsStage
from egoemg.models.heads.mlp import MLPHead
from egoemg.models.modules.emgformer import Emg2PoseFormer

SMALL_WINDOW_LENGTH = 12_000
SMALL_OUT_CHANNELS = 22
SMALL_CHANNEL_POSITIONS = np.asarray([10, 12, 0, 1, 2, 4, 5, 6], dtype=np.int64)


def build_small_emgformer(input_channels: int = 16) -> Emg2PoseFormer:
    featurizer = TdsNetwork(
        conv_blocks=[
            Conv1dBlock(
                in_channels=input_channels,
                out_channels=256,
                kernel_size=11,
                stride=5,
            ),
            Conv1dBlock(in_channels=256, out_channels=256, kernel_size=5, stride=2),
        ],
        tds_stages=[
            TdsStage(
                in_channels=256,
                in_conv_kernel_width=9,
                in_conv_stride=5,
                num_blocks=1,
                channels=8,
                feature_width=32,
                kernel_width=5,
            ),
            TdsStage(
                in_channels=256,
                in_conv_kernel_width=3,
                in_conv_stride=1,
                num_blocks=1,
                channels=8,
                feature_width=32,
                kernel_width=3,
            ),
        ],
        se={"enable": True, "reduction": 4, "residual": True, "mode": "global"},
    )
    decoder = TransformerDecoder(
        in_channels=256,
        model_dim=256,
        num_heads=4,
        num_layers=3,
        ffn_dim=512,
        dropout=0.1,
        activation="relu",
        norm_first=True,
        causal=False,
        pos_encoding="rope",
        out_proj=True,
    )
    head = MLPHead(
        in_channels=256,
        out_channels=SMALL_OUT_CHANNELS,
        hidden_sizes=[512],
        activation="relu",
        dropout=0.1,
    )
    return Emg2PoseFormer(
        featurizer=featurizer,
        decoder=decoder,
        head=head,
        out_channels=SMALL_OUT_CHANNELS,
        provide_initial_pos=False,
    )


def infer_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if key.endswith("featurizer.layers.0.conv.0.weight") or key == "featurizer.layers.0.conv.0.weight":
            return int(value.shape[1])
    for key, value in state_dict.items():
        if key.endswith("layers.0.conv.0.weight") and value.ndim == 3:
            return int(value.shape[1])
    return 16


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in checkpoint):
            return checkpoint
    raise ValueError("Unsupported checkpoint format")


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            stripped[key[len("model.") :]] = value
        elif key.startswith(("featurizer.", "decoder.", "head.")):
            stripped[key] = value
    return stripped


def load_small_emgformer(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    strict: bool = True,
) -> Emg2PoseFormer:
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path).expanduser()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        # Full Lightning checkpoints may contain OmegaConf objects that cannot be
        # loaded in minimal deployment envs. In training envs this fallback still
        # supports direct ckpt loading; deployment should prefer exported state
        # dict files made by scripts/realtime/export_small_runtime_checkpoint.py.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _strip_model_prefix(_extract_state_dict(ckpt))
    input_channels = infer_input_channels(state_dict)
    model = build_small_emgformer(input_channels=input_channels)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match local small EMGFormer architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval().to(target_device)
    return model


def map_small_channels(emg_8ch: np.ndarray) -> np.ndarray:
    """Place 8 hardware channels into the 16-channel training layout."""
    x = np.asarray(emg_8ch, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError(f"Expected (N, 8), got {x.shape}")
    return place_sparse_channels(x, 16, SMALL_CHANNEL_POSITIONS)


def prepare_small_input(emg_8ch: np.ndarray, model: Emg2PoseFormer) -> np.ndarray:
    input_channels = int(model.featurizer.layers[0].conv[0].weight.shape[1])
    if input_channels == 8:
        x = np.asarray(emg_8ch, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError(f"Expected (N, 8), got {x.shape}")
        return x
    if input_channels == 16:
        return map_small_channels(emg_8ch)
    raise ValueError(f"Unsupported small model input channel count: {input_channels}")
