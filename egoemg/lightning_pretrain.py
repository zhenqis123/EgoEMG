from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from egoemg import utils
from egoemg.keystroke_utils import (
    CTCGreedyDecoder,
    KeystrokeData,
    KeystrokeErrorRates,
)
from egoemg.datasets.emg2qwerty_dataset import CharacterSet


def _normalize_list(cfgs: Any) -> list[Any]:
    if cfgs is None:
        return []
    if OmegaConf.is_dict(cfgs):
        return [cfgs]
    if OmegaConf.is_list(cfgs):
        return list(cfgs)
    if isinstance(cfgs, Sequence) and not isinstance(cfgs, (str, bytes)):
        return list(cfgs)
    return [cfgs]


class EmgPretrainModule(pl.LightningModule):
    def __init__(
        self,
        module_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        gesture_spaces: Sequence[dict[str, Any]] | None = None,
        loss_weights: dict[str, float] | None = None,
        mask_conf: dict[str, Any] | None = None,
        recon_loss: str = "mse",
        angle_loss: str = "mae",
        label_smoothing: float = 0.0,
        datamodule: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(module_conf, _convert_="all")

        self.loss_weights = loss_weights or {
            "recon": 1.0,
            "angle": 1.0,
            "gesture": 1.0,
            "keystroke": 1.0,
        }
        self.recon_loss = recon_loss
        self.angle_loss = angle_loss
        self.label_smoothing = float(label_smoothing)

        mask_conf = mask_conf or {}
        self.mask_patch_size = int(mask_conf.get("patch_size", 50))
        self.mask_ratio = float(mask_conf.get("mask_ratio", 0.3))

        self.gesture_spaces = []
        total = 0
        self.gesture_offsets: list[int] = []
        for space in _normalize_list(gesture_spaces):
            if space is None:
                continue
            space_dict = dict(space)
            num_classes = int(space_dict["num_classes"])
            self.gesture_offsets.append(total)
            total += num_classes
            self.gesture_spaces.append(space_dict)
        self.gesture_num_classes = total

        if hasattr(self.model, "gesture_head") and hasattr(self.model.gesture_head, "net"):
            last_linear = None
            for layer in self.model.gesture_head.net:
                if isinstance(layer, torch.nn.Linear):
                    last_linear = layer
            if last_linear is not None and last_linear.out_features != total:
                raise ValueError(
                    f"gesture_head out_features ({last_linear.out_features}) "
                    f"!= gesture_num_classes ({total})"
                )

        # Initialize CTC components for keystroke supervision
        self.charset = CharacterSet()
        # CharacterSet uses allowed_keys to get the number of classes
        num_keystroke_classes = len(self.charset.allowed_keys)
        self.ctc_loss = torch.nn.CTCLoss(
            blank=num_keystroke_classes,  # Use last index as blank
            reduction='mean',
            zero_infinity=True
        )
        self.ctc_decoder = CTCGreedyDecoder(
            charset=self.charset,
            blank_idx=num_keystroke_classes
        )

        # Keystroke error rate metrics
        self.keystroke_metrics = torch.nn.ModuleDict({
            "train_keystroke_cer": KeystrokeErrorRates(),
            "val_keystroke_cer": KeystrokeErrorRates(),
        })

    _DETAIL_LOG_INTERVAL: int = 50

    def _should_log_detail(self) -> bool:
        return self.global_step % self._DETAIL_LOG_INTERVAL == 0

    def configure_optimizers(self):
        params = list(self.parameters())
        return utils.instantiate_optimizer_and_scheduler(
            params,
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )

    def _apply_batch_norm(self, batch: dict[str, torch.Tensor]) -> None:
        if (
            getattr(self.hparams, "datamodule", None)
            and self.hparams.datamodule.get("norm_mode") == "batch"
        ):
            emg = batch["emg"]
            mean = emg.mean()
            std = emg.std()
            batch["emg"] = (emg - mean) / (std + 1e-6)

    def _build_mask(self, emg: torch.Tensor) -> torch.Tensor:
        if self.mask_ratio <= 0.0 or self.mask_patch_size <= 0:
            return torch.zeros(emg.shape[0], emg.shape[-1], device=emg.device, dtype=torch.bool)

        b, _, t = emg.shape
        num_patches = math.ceil(t / self.mask_patch_size)
        n_mask = max(int(round(num_patches * self.mask_ratio)), 1)
        mask_patch = torch.zeros((b, num_patches), device=emg.device, dtype=torch.bool)
        for i in range(b):
            idx = torch.randperm(num_patches, device=emg.device)[:n_mask]
            mask_patch[i, idx] = True
        mask_time = mask_patch.repeat_interleave(self.mask_patch_size, dim=1)
        return mask_time[:, :t]

    def _mask_emg(self, emg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return emg
        # mean = emg.mean(dim=-1, keepdim=True)
        return torch.where(mask[:, None, :], torch.zeros_like(emg), emg)

    @staticmethod
    def _align(pred: torch.Tensor, n_time: int) -> torch.Tensor:
        if pred.shape[-1] == n_time:
            return pred
        return F.interpolate(pred, size=n_time, mode="linear")

    def _masked_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str
    ) -> torch.Tensor:
        if mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = pred - target
        if loss_type == "mae":
            diff = diff.abs()
        elif loss_type == "mse":
            diff = diff.pow(2)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.dim() == 2:
            mask = mask[:, None, :]
        denom = mask.sum().clamp(min=1).to(diff.dtype)
        if mask.shape[1] == 1 and diff.dim() == 3 and diff.shape[1] > 1:
            denom = denom * diff.shape[1]
        return (diff * mask).sum() / denom

    def _gesture_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        if self.gesture_num_classes == 0:
            return torch.tensor(0.0, device=logits.device)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        count = 0
        space_losses: dict[str, torch.Tensor] = {}
        space_accs: dict[str, float] = {}
        for i, space in enumerate(self.gesture_spaces):
            space_name = space.get("name", f"space_{i}")
            offset = self.gesture_offsets[i]
            num_classes = int(space["num_classes"])
            mask = masks[:, i, :]
            if mask.sum() == 0:
                space_losses[space_name] = torch.tensor(0.0, device=logits.device)
                space_accs[space_name] = 0.0
                continue
            logits_space = logits[:, offset : offset + num_classes, :]
            logits_flat = logits_space.permute(0, 2, 1).reshape(-1, num_classes)
            labels_flat = labels[:, i, :].reshape(-1)
            mask_flat = mask.reshape(-1).bool()

            # Calculate loss
            loss = F.cross_entropy(
                logits_flat[mask_flat],
                labels_flat[mask_flat],
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + loss

            # Calculate accuracy
            preds = logits_flat[mask_flat].argmax(dim=-1)
            correct = (preds == labels_flat[mask_flat]).sum()
            n_correct = correct.item()
            n_samples = mask_flat.sum().item()
            total_correct += n_correct
            total_samples += n_samples

            space_losses[space_name] = loss
            space_accs[space_name] = n_correct / n_samples if n_samples > 0 else 0.0

            count += 1

        # Log per-space metrics only every N steps to reduce overhead
        if self._should_log_detail():
            for space_name in space_losses:
                self.log(
                    f"{stage}_gesture_ce/{space_name}", space_losses[space_name],
                    sync_dist=True, batch_size=logits.shape[0],
                )
                self.log(
                    f"{stage}_gesture_acc/{space_name}", space_accs[space_name],
                    sync_dist=True, batch_size=logits.shape[0],
                )

        if count == 0:
            return torch.tensor(0.0, device=logits.device)

        loss = total_loss / count
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        self.log(f"{stage}_gesture_ce", loss, sync_dist=True, batch_size=logits.shape[0])
        self.log(f"{stage}_gesture_acc", accuracy, sync_dist=True, batch_size=logits.shape[0])
        return loss

    def _keystroke_loss(
        self,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        stage: str,
    ) -> tuple[torch.Tensor, bool]:
        """Compute CTC loss for keystroke sequences.

        Returns:
            (loss, has_data): loss tensor and whether any valid keystroke data exists
        """
        keystroke_labels = batch.get("keystroke_labels")
        if keystroke_labels is None:
            # Debug: print once per epoch
            if not hasattr(self, '_keystroke_debug_no_labels'):
                self._keystroke_debug_no_labels = True
                print(f"[DEBUG] No keystroke_labels in batch")
            return torch.tensor(0.0, device=batch["emg"].device), False
        
        if not hasattr(self.model, "keystroke_head"):
            if not hasattr(self, '_keystroke_debug_no_head'):
                self._keystroke_debug_no_head = True
                print(f"[DEBUG] Model has no keystroke_head attribute")
            return torch.tensor(0.0, device=batch["emg"].device), False

        # Check if any samples have keystroke labels
        has_labels = any(len(labels) > 0 for labels in keystroke_labels)
        if not has_labels:
            if not hasattr(self, '_keystroke_debug_empty_labels'):
                self._keystroke_debug_empty_labels = True
                print(f"[DEBUG] All keystroke_labels are empty")
            return torch.tensor(0.0, device=batch["emg"].device), False
        
        # Get keystroke logits from model (already in T, B, C format)
        keystroke_logits = outputs.get("keystroke_logits")
        if keystroke_logits is None:
            if not hasattr(self, '_keystroke_debug_no_logits'):
                self._keystroke_debug_no_logits = True
                print(f"[DEBUG] No keystroke_logits in model outputs")
                print(f"[DEBUG] Available keys: {list(outputs.keys())}")
                print(f"[DEBUG] Model has keystroke_head: {self.model.keystroke_head is not None}")
            return torch.tensor(0.0, device=batch["emg"].device), False

        # keystroke_logits is (T, B, C), slice time dimension for context trimming
        if keystroke_logits.dim() != 3:
            return torch.tensor(0.0, device=batch["emg"].device), False

        # keystroke_logits is already in the correct time dimension after featurizer/decoder
        # No need to slice again - the featurizer already accounts for left/right context
        # Unlike other tasks where we slice the target, CTC loss works with the full sequence
        T, N, _ = keystroke_logits.shape

        if T == 0:
            return torch.tensor(0.0, device=batch["emg"].device), False
        # Prepare targets and lengths
        max_target_len = max(len(labels) for labels in keystroke_labels)
        targets = torch.full((N, max_target_len), 0, dtype=torch.long, device=keystroke_logits.device)
        target_lengths = torch.zeros(N, dtype=torch.long, device=keystroke_logits.device)

        for i, labels in enumerate(keystroke_labels):
            if len(labels) > 0:
                targets[i, :len(labels)] = labels
                target_lengths[i] = len(labels)

        # Input lengths (all same for this batch)
        input_lengths = torch.full((N,), T, dtype=torch.long, device=keystroke_logits.device)

        # Compute CTC loss
        log_probs = torch.log_softmax(keystroke_logits, dim=-1)

        # Only compute loss for samples with labels
        valid_mask = target_lengths > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=batch["emg"].device), False

        # Flatten targets for CTCLoss
        targets_flat = torch.cat([targets[i, :target_lengths[i]] for i in range(N) if valid_mask[i]])
        
        loss = self.ctc_loss(
            log_probs[:, valid_mask, :],
            targets_flat,
            input_lengths[valid_mask],
            target_lengths[valid_mask],
        )

        # Decode and compute error rates (for both train and val/test)
        with torch.no_grad():
            emissions = log_probs.detach().cpu().numpy()
            emission_lengths = input_lengths.detach().cpu().numpy()
            predictions = self.ctc_decoder.decode_batch(emissions, emission_lengths)

            for i in range(N):
                if valid_mask[i]:
                    target_data = KeystrokeData.from_labels(
                        keystroke_labels[i].cpu().numpy(), self.charset
                    )
                    self.keystroke_metrics[f"{stage}_keystroke_cer"].update(
                        predictions[i], target_data
                    )

        return loss, True

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        self._apply_batch_norm(batch)

        emg = batch["emg"]
        mask = self._build_mask(emg)
        emg_masked = self._mask_emg(emg, mask)

        outputs = self.model({"emg": emg_masked})

        start = getattr(self.model, "left_context", 0)
        stop = None if getattr(self.model, "right_context", 0) == 0 else -self.model.right_context

        batch_size = emg.shape[0]
        w_recon = self.loss_weights.get("recon", 0.0)
        w_angle = self.loss_weights.get("angle", 0.0)
        w_gesture = self.loss_weights.get("gesture", 0.0)
        w_keystroke = self.loss_weights.get("keystroke", 0.0)
        zero = torch.tensor(0.0, device=emg.device)

        # Recon loss
        if w_recon > 0 and "recon" in outputs:
            target_emg = emg[..., slice(start, stop)]
            target_mask = mask[..., slice(start, stop)]
            recon_pred = self._align(outputs["recon"], target_emg.shape[-1])
            recon_loss = self._masked_loss(recon_pred, target_emg, target_mask, self.recon_loss)
        else:
            recon_loss = zero

        # Angle loss
        if w_angle > 0 and "angles" in outputs:
            angle_target = batch["angle_target"][..., slice(start, stop)]
            angle_mask = batch["angle_mask"][..., slice(start, stop)]
            angle_pred = self._align(outputs["angles"], angle_target.shape[-1])
            angle_loss = self._masked_loss(angle_pred, angle_target, angle_mask, self.angle_loss)

            # Per-dataset angle loss
            # Always log during val/test; only gate by interval during training
            log_detail = self._should_log_detail() or stage != "train"
            dataset_names = batch.get("dataset_name")
            if dataset_names is not None and log_detail:
                for ds_name in ("emg2pose", "pimforce", "ninapro", "egoemg"):
                    ds_mask_bool = torch.tensor(
                        [n == ds_name for n in dataset_names],
                        dtype=torch.bool, device=angle_pred.device,
                    )
                    if ds_mask_bool.any():
                        ds_pred = angle_pred[ds_mask_bool]
                        ds_target = angle_target[ds_mask_bool]
                        ds_amask = angle_mask[ds_mask_bool]
                        if ds_amask.any():
                            ds_loss = self._masked_loss(ds_pred, ds_target, ds_amask, self.angle_loss)
                            self.log(
                                f"{stage}_angle_loss/{ds_name}", ds_loss,
                                sync_dist=True, batch_size=ds_mask_bool.sum().item(),
                            )
        else:
            angle_loss = zero

        # Gesture loss
        if w_gesture > 0 and "gesture_logits" in outputs:
            gesture_labels = batch["gesture_labels"][..., slice(start, stop)]
            gesture_masks = batch["gesture_masks"][..., slice(start, stop)]
            gesture_logits = self._align(outputs["gesture_logits"], gesture_labels.shape[-1])
            gesture_loss = self._gesture_loss(gesture_logits, gesture_labels, gesture_masks, stage)
        else:
            gesture_loss = zero

        # Keystroke CTC loss (only for emg2qwerty data)
        if w_keystroke > 0:
            keystroke_loss, keystroke_has_data = self._keystroke_loss(
                batch, outputs, stage
            )
        else:
            keystroke_loss = zero

        total = (
            recon_loss * w_recon
            + angle_loss * w_angle
            + gesture_loss * w_gesture
            + keystroke_loss * w_keystroke
        )

        self.log(f"{stage}_loss", total, sync_dist=True, batch_size=batch_size)
        if w_recon > 0 and recon_loss is not zero:
            self.log(f"{stage}_recon_loss", recon_loss, sync_dist=True, batch_size=batch_size)
        if w_angle > 0:
            self.log(f"{stage}_angle_loss", angle_loss, sync_dist=True, batch_size=batch_size)
        if w_gesture > 0:
            self.log(f"{stage}_gesture_ce", gesture_loss, sync_dist=True, batch_size=batch_size)
        if w_keystroke > 0 and keystroke_loss is not zero:
            self.log(f"{stage}_keystroke_loss", keystroke_loss, sync_dist=True, batch_size=batch_size)

        return total

    def training_step(self, batch, _batch_idx) -> torch.Tensor:
        loss = self._step(batch, stage="train")

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._step(batch, stage="test")

    def on_train_epoch_end(self) -> None:
        """Log keystroke error rates at end of training epoch."""
        if self.loss_weights.get("keystroke", 0.0) > 0 and "train_keystroke_cer" in self.keystroke_metrics:
            metrics = self.keystroke_metrics["train_keystroke_cer"].compute()
            for key, value in metrics.items():
                self.log(f"train_keystroke_{key}", value, sync_dist=True)
            self.keystroke_metrics["train_keystroke_cer"].reset()

    def on_validation_epoch_end(self) -> None:
        """Log keystroke error rates at end of validation epoch."""
        if self.loss_weights.get("keystroke", 0.0) > 0 and "val_keystroke_cer" in self.keystroke_metrics:
            metrics = self.keystroke_metrics["val_keystroke_cer"].compute()
            for key, value in metrics.items():
                self.log(f"val_keystroke_{key}", value, sync_dist=True)
            self.keystroke_metrics["val_keystroke_cer"].reset()

    def on_test_epoch_end(self) -> None:
        """Log keystroke error rates at end of test epoch."""
        if self.loss_weights.get("keystroke", 0.0) > 0 and "test_keystroke_cer" in self.keystroke_metrics:
            metrics = self.keystroke_metrics["test_keystroke_cer"].compute()
            for key, value in metrics.items():
                self.log(f"test_keystroke_{key}", value, sync_dist=True)
            self.keystroke_metrics["test_keystroke_cer"].reset()
