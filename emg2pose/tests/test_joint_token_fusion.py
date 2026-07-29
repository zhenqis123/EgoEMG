import torch

from emg2pose.models.modules.mid_fusion import JointTokenFusionEncoder


def test_joint_token_fusion_encoder_returns_one_pose_token_per_sample() -> None:
    encoder = JointTokenFusionEncoder(
        emg_dim=256,
        vision_dim=512,
        token_dim=256,
        num_heads=8,
        num_layers=2,
        ffn_dim=512,
        dropout=0.0,
    )
    emg_features = torch.randn(2, 256, 230, requires_grad=True)
    vision_map = torch.randn(2, 512, 8, 8, requires_grad=True)

    pose_feature = encoder(
        emg_features,
        vision_map,
        vision_valid=torch.tensor([True, False]),
    )

    assert pose_feature.shape == (2, 256)
    pose_feature.square().mean().backward()
    assert emg_features.grad is not None
    assert vision_map.grad is not None
