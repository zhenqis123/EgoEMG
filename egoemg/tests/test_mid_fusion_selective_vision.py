import torch

from egoemg.models.modules.mid_fusion import MidFusionPoseFormer


def test_selective_vision_lock_survives_train_mode() -> None:
    model = object.__new__(MidFusionPoseFormer)
    torch.nn.Module.__init__(model)
    model.freeze_vision_branch = False
    model.freeze_vision_head = True
    model.lock_vision_batch_norm = True
    model.vision_backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.BatchNorm2d(4),
    )
    model.head_vision = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4, 2),
    )

    model.train()

    assert model.training
    assert model.vision_backbone[0].training
    assert not model.vision_backbone[1].training
    assert not model.head_vision.training
