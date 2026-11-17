"""Train-only fused-angle MAE with separate validation and test evaluation."""

import torch

from egoemg.training.prediction import EmgPredictionModule


class FeatureGatePredictionModule(EmgPredictionModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (all + four splits) x (fusion error, vision error, count)
        self.register_buffer('_gate_eval_sums', torch.zeros(5, 3,
                             dtype=torch.float64), persistent=False)
        self.register_buffer('_gate_valid_emg_sums', torch.zeros(3,
                             dtype=torch.float64), persistent=False)

    def _gate_step(self, batch):
        features = self.model.encode_experts(batch)
        prediction, _ = self.model.forward_cached(features)
        target = batch['joint_angles'].reshape_as(prediction).float()
        valid = batch['label_valid_mask'].reshape_as(prediction).bool()
        error = (prediction-target).abs().masked_fill(~valid, 0.)
        loss = error.sum()/valid.sum().clamp_min(1)
        return loss, prediction, target, valid, features

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self._gate_step(batch)
        size = len(batch['emg'])
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=size, prog_bar=True)
        self.log('train_mae', loss.detach(), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=size)
        return loss

    def on_validation_epoch_start(self):
        self._gate_eval_sums.zero_()
        self._gate_valid_emg_sums.zero_()

    def _evaluation_step(self, batch):
        loss, pred, target, valid, features = self._gate_step(batch)
        error = (pred-target).abs().double()
        vision_error = (features['vision_prediction']-target).abs().double()
        if 'emg_valid_mask' in batch:
            mask = valid & batch['emg_valid_mask'].reshape(-1, 2, 1).bool()
            self._gate_valid_emg_sums += torch.stack((
                error.masked_fill(~mask, 0.).sum(),
                vision_error.masked_fill(~mask, 0.).sum(), mask.sum(),
            ))
        for code in range(5):
            mask = valid if code == 0 else (
                valid & (batch['split_id'].reshape(-1, 2) == code)[..., None])
            sums = self._gate_eval_sums[code]
            sums[0] += error.masked_fill(~mask, 0.).sum()
            sums[1] += vision_error.masked_fill(~mask, 0.).sum()
            sums[2] += mask.sum()
        return loss

    def _finish_gate_metrics(self, role):
        sums = self._gate_eval_sums.clone()
        valid_emg_sums = self._gate_valid_emg_sums.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(valid_emg_sums)
        names = ['all', 'user', 'gesture', 'both', 'recent_object']
        for code, name in enumerate(names):
            fusion, vision, count = sums[code]
            if count > 0:
                suffix = '' if code == 0 else '_'+name
                self.log(role+suffix+'_mae', fusion/count, sync_dist=False,
                         add_dataloader_idx=False, prog_bar=code == 0)
                self.log(role+suffix+'_vision_mae', vision/count,
                         sync_dist=False, add_dataloader_idx=False)
        emg_sums = valid_emg_sums
        if emg_sums[2] > 0:
            for name, value in [('mae', emg_sums[0]),
                                ('vision_mae', emg_sums[1])]:
                self.log(role+'_valid_emg_'+name, value/emg_sums[2],
                         sync_dist=False, add_dataloader_idx=False)
            self.log(role+'_valid_emg_fraction', emg_sums[2]/sums[0, 2],
                     sync_dist=False, add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch)

    def on_validation_epoch_end(self):
        self._finish_gate_metrics('val')

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch)

    def on_test_epoch_end(self):
        self._finish_gate_metrics('test')


class ResidualFusionPredictionModule(FeatureGatePredictionModule):
    """Train residual fusion with final-angle MAE."""

    def _gate_step(self, batch):
        prediction, target, valid = self.model(batch)
        prediction = prediction.reshape(-1, 2, 22).float()
        target = target.reshape_as(prediction).float()
        valid = valid.reshape_as(prediction).bool()
        loss = (prediction-target).abs().masked_fill(~valid, 0.).sum()
        loss = loss/valid.sum().clamp_min(1)
        features = dict(vision_prediction=self.model._last_vision_prediction.reshape(
            -1, 2, 22).float())
        return loss, prediction, target, valid, features


class VisionPredictionModule(FeatureGatePredictionModule):
    """Vision training with final-angle MAE."""

    def _gate_step(self, batch):
        prediction, target, valid = self.model(batch)
        prediction = prediction.reshape(-1, 2, 22).float()
        target = target.reshape_as(prediction).float()
        valid = valid.reshape_as(prediction).bool()
        loss = (prediction-target).abs().masked_fill(~valid, 0.).sum()
        loss = loss/valid.sum().clamp_min(1)
        return loss, prediction, target, valid, dict(vision_prediction=prediction)
