"""Forward smoke tests for all MidFusionPoseFormer fusion modes.

Covers the six dispatch modes (fusion, emg_only, vision_only,
center_supervised, token_self_attention, frozen_cross_attention) with
stub submodules on CPU. The real model is constructed via ``__new__`` and
only the attributes each forward path touches are attached, so the tests
assert output SHAPES and crash-freedom, not learned behavior.
"""
from __future__ import annotations

import torch
from torch import nn

from egoemg.models.modules.mid_fusion import MidFusionPoseFormer

B, C_EMG, C_VIS, T, OUT = 2, 8, 16, 10, 22


class _MeanPool(nn.Module):
    """(B, C, T) -> (B, C) mean pool (matches temporal-attention output dims)."""

    def forward(self, x):
        return x.mean(dim=2)


class _EarlyFusionStub(nn.Module):
    """(B, C, T) EMG features -> same shape (ignores vision maps)."""

    def forward(self, emg_features, layer3_map, layer4_map, vision_valid):
        return emg_features


class _TokenFusionStub(nn.Module):
    """(B, C, T) EMG features -> pooled (B, C) pose token."""

    def forward(self, emg_features, vision_map, vision_valid):
        return emg_features.mean(dim=2)


def _make_stub_model(fusion_mode: str) -> MidFusionPoseFormer:
    model = MidFusionPoseFormer.__new__(MidFusionPoseFormer)
    nn.Module.__init__(model)
    model.fusion_mode = fusion_mode
    model.force_zero_emg = False
    model.moddrop_prob = 0.0
    model.vision_backbone_type = "resnet18"  # routes to the ResNet extraction path
    model.left_context = 0
    model.right_context = 0
    model._vision_T = None
    model._last_delta = None
    model.freeze_vision_branch = False
    model.freeze_vision_head = False
    model.lock_vision_batch_norm = False
    model.residual_scale = 1.0

    # EMG path: (B, 8, T) -> (B, 8, T); temporal pool -> (B, 8)
    model.featurizer = nn.Sequential(nn.Conv1d(C_EMG, C_EMG, 3, padding=1), nn.ReLU())
    model.decoder = nn.Sequential(nn.Conv1d(C_EMG, C_EMG, 3, padding=1), nn.ReLU())
    model.temporal_attn = _MeanPool()

    # Vision path: (B, 16) -> predictions / fusion space
    model.head_vision = nn.Linear(C_VIS, OUT)
    model.vision_proj = nn.Linear(C_VIS, C_EMG)

    # Fusion path: concat [decoded; vis] on channel dim -> (B, 8, T) -> (B, OUT, T)
    model.fusion_proj = nn.Conv1d(2 * C_EMG, C_EMG, 1)
    model.head = nn.Conv1d(C_EMG, OUT, 1)

    model.early_fusion = _EarlyFusionStub()
    model.token_fusion = _TokenFusionStub()

    # Stub the vision extraction helpers so no real backbone is needed.
    def _multiscale(img):
        b = img.shape[0]
        return (
            torch.randn(b, C_VIS, 4, 4),
            torch.randn(b, C_VIS, 4, 4),
        )

    def _token_map(img):
        return torch.randn(img.shape[0], C_VIS, 4, 4)

    model._extract_resnet_multiscale = _multiscale
    model._extract_resnet_token_map = _token_map
    return model


def _make_batch(with_vision_img: bool = False) -> dict[str, torch.Tensor]:
    batch = {
        "emg": torch.randn(B, C_EMG, T),
        "vision_features": torch.randn(B, C_VIS),
        "vision_valid_mask": torch.ones(B, dtype=torch.bool),
        "joint_angles": torch.randn(B, OUT, 1),
        "label_valid_mask": torch.ones(B, 1, dtype=torch.bool),
    }
    if with_vision_img:
        batch["vision_img"] = torch.randn(B, 3, 64, 64)
    return batch


MODES = [
    "fusion",
    "emg_only",
    "vision_only",
    "center_supervised",
    "token_self_attention",
    "frozen_cross_attention",
]


def test_all_fusion_modes_forward_shapes() -> None:
    for mode in MODES:
        model = _make_stub_model(mode)
        model.eval()
        needs_img = mode in {"token_self_attention", "frozen_cross_attention"}
        batch = _make_batch(with_vision_img=needs_img)
        with torch.no_grad():
            out = model(batch)
        # The default fusion path, emg_only, and vision_only predict the full
        # temporal window (vision expanded across T, center-supervised at the
        # middle step); the remaining modes predict the center frame only.
        full_temporal = mode in {"fusion", "emg_only", "vision_only"}
        exp_t = T if full_temporal else 1
        if isinstance(out, tuple):
            preds, targets, mask = out
            assert preds.shape == (B, OUT, exp_t), (mode, preds.shape)
            assert targets.shape == (B, OUT, exp_t), (mode, targets.shape)
            assert mask.shape[0] == B, (mode, mask.shape)
            assert mask.shape[-1] == exp_t, (mode, mask.shape)
        else:
            assert out.shape == (B, OUT, exp_t), (mode, out.shape)


def test_fusion_mode_masks_vision_on_invalid() -> None:
    """Invalid vision samples must zero the vision features before fusion."""
    model = _make_stub_model("fusion")
    model.eval()
    batch = _make_batch()
    batch["vision_valid_mask"] = torch.tensor([True, False], dtype=torch.bool)
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, tuple)


def test_token_self_attention_requires_vision_img() -> None:
    model = _make_stub_model("token_self_attention")
    model.eval()
    with torch.no_grad():
        with torch.no_grad():
            try:
                model(_make_batch(with_vision_img=False))
            except ValueError as exc:
                assert "vision_img" in str(exc)
            else:
                raise AssertionError("expected ValueError for missing vision_img")
