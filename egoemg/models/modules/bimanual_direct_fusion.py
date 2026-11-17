"""Matched feature-fusion control without an angle-space residual addition."""

from contextlib import nullcontext

import torch

from egoemg.models.modules.bimanual_simple_residual import BimanualSimpleResidualFusion


class BimanualDirectFusion(BimanualSimpleResidualFusion):
    """Reuse residual features/projections/heads, directly regress 22 angles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep identical shared-layer initialization, but initialize direct
        # regression outputs normally instead of near-zero corrections.
        heads = self.head if self.independent_hand_heads else [self.head]
        with torch.random.fork_rng(devices=[]):
            for head in heads:
                head.net[-1].reset_parameters()
        self.vision.head.requires_grad_(False)

    def forward(self, batch):
        self._validate_causal_batch(batch)
        image = batch["vision_img"]
        with torch.no_grad() if self.freeze_vision else nullcontext():
            if hasattr(self.vision, "forward_features"):
                visual = self.vision.forward_features(batch)
            else:
                if image.dtype == torch.uint8:
                    image = (image.float().div(255).sub(self.vision.image_mean)
                             .div(self.vision.image_std))
                visual = self.vision.backbone(image)
        # Diagnostic only: never add this prediction to the direct output.
        with torch.no_grad():
            self._last_vision_prediction = self.vision.head(visual).reshape(-1, 22, 1)
        signal = batch["emg"].reshape(-1, 8, batch["emg"].shape[-1])
        available = batch["emg_valid_mask"].reshape(-1).bool()
        indices = available.nonzero(as_tuple=True)[0]
        pooled = visual.new_zeros((len(signal), self.emg_dim))
        if indices.numel():
            with torch.no_grad() if self.freeze_emg else nullcontext():
                features = self.emg_model.decoder(self.emg_model.featurizer(signal[indices]))
            pooled = pooled.index_copy(0, indices, self._pool_emg(features).to(pooled.dtype))
        hand_id = torch.eye(2, device=visual.device, dtype=visual.dtype).repeat(
            len(image), 1
        )
        prediction = self._predict_delta(visual, pooled, hand_id)
        # Missing EMG stays a zero feature, not a zero angle or vision fallback.
        if "joint_angles" not in batch:
            return prediction
        return (
            prediction,
            batch["joint_angles"].reshape(-1, 22, 1),
            batch["label_valid_mask"].reshape(-1, 22, 1).bool(),
        )
