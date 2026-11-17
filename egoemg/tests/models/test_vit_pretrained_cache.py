"""Local timm pretraining must not require an online cache freshness check."""

from types import SimpleNamespace

import huggingface_hub
import pytest
import timm
from torch import nn

from egoemg.models.modules.vit_vision import VisionViTPose


def assert_cached_initialization(monkeypatch, tmp_path, build_model):
    monkeypatch.setenv("EGOEMG_NO_PRETRAINED_DOWNLOAD", "0")
    cached = tmp_path / "model.safetensors"
    cached.write_bytes(b"mock")
    monkeypatch.setattr(
        timm.models,
        "get_pretrained_cfg",
        lambda name: SimpleNamespace(hf_hub_id="timm/test"),
    )
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda repo, filename: str(cached)
    )
    calls = []

    def create(name, **kwargs):
        calls.append(kwargs)
        return nn.Identity()

    monkeypatch.setattr(timm, "create_model", create)
    build_model()
    assert calls[0]["pretrained"] is True
    assert calls[0]["pretrained_cfg_overlay"] == {"file": str(cached)}
    assert calls[0]["img_size"] == 256


def test_vit_uses_cached_file_with_timm_filter(monkeypatch, tmp_path):
    assert_cached_initialization(
        monkeypatch, tmp_path, lambda: VisionViTPose(backbone_type="vit_small")
    )


def test_explicit_missing_pretrained_path_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        VisionViTPose(
            backbone_type="vit_small",
            pretrained_path=str(tmp_path / "missing.safetensors"),
        )
