"""Session RGB bytes and legacy normalized images must predict the same angles."""
import torch

from egoemg.models.modules.resnet_vision import ResNetVisionPose


def test_resnet_uint8_matches_normalized_float():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = ResNetVisionPose(pretrained=False).eval()
    image = torch.randint(256, (2, 3, 32, 32), dtype=torch.uint8)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    normalized = (image.float() / 255 - mean) / std
    with torch.no_grad():
        actual = model({'vision_img': image})
        expected = model({'vision_img': normalized})
    assert actual.shape == (2, 22, 1)
    torch.testing.assert_close(actual, expected)
    assert 'image_mean' not in model.state_dict()
