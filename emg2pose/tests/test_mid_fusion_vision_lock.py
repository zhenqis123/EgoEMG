from __future__ import annotations

from torch import nn

from emg2pose.models.modules.mid_fusion import MidFusionPoseFormer


def test_protected_vision_branch_stays_eval_when_model_trains() -> None:
    """A protected branch must not update BatchNorm or activate dropout."""
    model = MidFusionPoseFormer.__new__(MidFusionPoseFormer)
    nn.Module.__init__(model)
    model.freeze_vision_branch = True
    model.vision_backbone = nn.Sequential(nn.BatchNorm2d(2), nn.ReLU())
    model.head_vision = nn.Sequential(nn.Dropout(), nn.Linear(2, 2))

    model.train()

    assert model.training
    assert not model.vision_backbone.training
    assert not model.head_vision.training
