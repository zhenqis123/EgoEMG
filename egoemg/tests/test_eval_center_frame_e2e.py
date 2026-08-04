"""End-to-end smoke test for the unified center-frame evaluator.

Builds a synthetic memmap large enough to host REF_WL=7790 grid windows,
monkeypatches the checkpoint loader and CUDA (CPU-only CI), and asserts the
per-hand MAE computation over the reference grid. A stub model that returns
its targets exercises the full loop: dataset construction, per-center
sampling, validity gating, and metric aggregation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egoemg import center_frame
from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from egoemg.tests.test_egoemg_memmap_dataset import _build_synthetic_memmap

REF_WL = center_frame.REF_WL


class _StubNet(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Perfect predictor: return the targets unchanged -> MAE == 0.
        return batch["joint_angles"]


class _StubModule:
    """Minimal stand-in for the Lightning module (no .cuda() on CPU CI)."""

    def __init__(self) -> None:
        self.model = _StubNet()

    def cuda(self):
        return self

    def eval(self):
        return self


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "datamodule": {
                "window_length": 500,
                "norm_mode": "none",
                "per_dataset_norm_stats_path": None,
                # Mirrors the real resolved configs (dataset_conf.val.0 is the
                # legacy dataset template OmegaConf.select falls back to).
                "dataset_conf": {"val": [{}]},
            },
            "egoemg_emg_layout": "target_hand",
            "egoemg_emg2pose_channel_indices": None,
            "egoemg_channel_interpolate": False,
            "skip_emg_loading": False,
            "center_target_only": True,
            "vision_num_frames": 1,
            "per_episode_crops_dir": None,
            "vision_patch_size": 256,
        }
    )


def test_eval_center_frame_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # CUDA is a no-op so the eval loop runs on CPU-only CI.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(center_frame, "_load_module", lambda cfg, ckpt: _StubModule())
    # Synthetic memmap has no real pre-crop LMDB or videos; hand back a dummy
    # patch so the vision-enabled center_supervised path runs end to end.
    monkeypatch.setattr(
        EgoEmgMemmapDataset,
        "_read_episode_crops",
        lambda self, ep_id, frame_idxs, hand_code: np.zeros(
            (1, 256, 256, 3), dtype=np.uint8
        ),
    )

    mm = _build_synthetic_memmap(tmp_path / "mm", n_rows=2 * REF_WL + 500, n_episodes=1)
    # The fixture marks every frame as train; flip to val so the grid has centers.
    n_rows = 2 * REF_WL + 500
    split = np.memmap(mm / "frame_split_id.dat", dtype="int8", mode="r+", shape=(n_rows,))
    split[:] = 1
    split.flush()

    results = center_frame.eval_center_frame(
        _make_cfg(), "dummy.ckpt", mm, center_window_length=500
    )

    assert set(results.keys()) == {"left", "right"}
    for hand, r in results.items():
        # (2 * REF_WL + 500 - REF_WL) // REF_WL + 1 == 2 grid windows
        assert r["n_valid"] == 2, (hand, r)
        assert r["overall_mae"] == 0.0, (hand, r)
        assert len(r["per_joint"]) == 22, (hand, r)
