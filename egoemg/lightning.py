# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import time

from collections.abc import Mapping
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only

from egoemg import utils
from egoemg.metrics import get_default_metrics
from egoemg.models.modules import BaseModule
from egoemg.transforms_batch import BatchAugmentation
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _debug_steps_enabled() -> bool:
    return os.environ.get("EMG2POSE_DEBUG_STEPS", "0").lower() in {"1", "true", "yes"}


def _load_state_dict_from_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load state_dict from checkpoint file, handling various formats."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    # These experiment checkpoints are trusted local artifacts and include
    # OmegaConf objects in their saved metadata.  PyTorch 2.6+ defaults to
    # ``weights_only=True``, which rejects that metadata before the state dict
    # can be extracted.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        else:
            return checkpoint
    return checkpoint


class EmgPredictionModule(pl.LightningModule):
    def __init__(
        self,
        module_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        loss_weights: dict[str, float] | None = None,
        pretrained_checkpoint: str | None = None,
        pretrained_strict: bool = False,
        pretrained_emg_checkpoint: str | None = None,
        freeze_backbone: bool = False,
        ignore_head_tail_dims: int = 0,
        datamodule: DictConfig | None = None,
        stage2_vision_checkpoint: str | None = None,
        component_lr_scales: dict[str, float] | None = None,
        batch_augmentation: DictConfig | None = None,
        val_episode_name_mapping: dict[str, str] | None = None,
        anchor_loss_weight: float = 0.0,
        anchor_shuffle_fraction: float = 0.0,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.model: BaseModule = instantiate(module_conf, _convert_="all")
        self.loss_weights = loss_weights or {"mae": 1}
        self.ignore_head_tail_dims = int(ignore_head_tail_dims)
        self.component_lr_scales = component_lr_scales or {}
        self.anchor_loss_weight = float(anchor_loss_weight)
        self.anchor_shuffle_fraction = float(anchor_shuffle_fraction)
        if not 0.0 <= self.anchor_shuffle_fraction <= 1.0:
            raise ValueError("anchor_shuffle_fraction must be in [0, 1]")
        self.task_type = "regression"

        # Batch augmentation on GPU (replaces per-sample CPU transforms)
        if batch_augmentation is not None:
            self.batch_aug = BatchAugmentation(
                OmegaConf.to_container(batch_augmentation, resolve=True)
            )
        else:
            self.batch_aug = None

        # Metrics sets
        self.regression_metrics = get_default_metrics()

        # Per-split validation MAE accumulators
        # Tracks MAE separately for each generalization condition:
        #   "stage" (seen user, unseen gesture),
        #   "user" (unseen user, seen gesture),
        #   "user_stage" (unseen user + unseen gesture)
        self._val_split_sums: dict[str, float] = {}
        self._val_split_counts: dict[str, float] = {}

        # Per-episode validation MAE accumulators
        self._val_episode_sums: dict[str, float] = {}
        self._val_episode_counts: dict[str, float] = {}
        self.val_episode_name_mapping: dict[str, str] = val_episode_name_mapping or {}

        # Per-joint validation MAE accumulators
        self._val_joint_sums: dict[int, float] = {}
        self._val_joint_counts: dict[int, float] = {}

        if pretrained_checkpoint is not None:
            self._load_pretrained_backbone(
                pretrained_checkpoint, strict=bool(pretrained_strict)
            )
            self._load_pretrained_angle_head(
                pretrained_checkpoint, strict=bool(pretrained_strict)
            )

        if pretrained_emg_checkpoint is not None:
            self._load_pretrained_backbone(
                pretrained_emg_checkpoint, strict=bool(pretrained_strict)
            )

        if stage2_vision_checkpoint is not None:
            self._load_fusion_vision_weights(stage2_vision_checkpoint)

        if freeze_backbone:
            self._freeze_backbone()

        # Apply component-level freezing for scale=0 entries
        for comp, scale in self.component_lr_scales.items():
            if scale == 0.0:
                self._freeze_component(comp)

    def _debug_step_log(self, message: str) -> None:
        if not _debug_steps_enabled():
            return
        rank = getattr(self.trainer, "global_rank", 0) if self.trainer is not None else 0
        print(f"[emg2pose-debug][rank={rank}] {message}", flush=True)

    @staticmethod
    def _debug_sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._log_param_breakdown()

    def _load_pretrained_backbone(self, checkpoint_path: str, strict: bool = False) -> None:
        state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

        model_state = self.model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key[6:] if key.startswith("model.") else key
            if not stripped.startswith((
                "featurizer.", "decoder.", "backbone.", "avgpool.",
                "vision_backbone.", "vision_proj.", "fusion_proj.", "head_vision.",
                "temporal_attn.",  # EMG temporal attention pooling (center_supervised fusion)
                "token_fusion.",  # joint EMG/vision token Transformer
                "early_fusion.",  # frozen multi-scale visual cross-attention
            )):
                continue
            if stripped in model_state and model_state[stripped].shape == value.shape:
                filtered[stripped] = value

        missing_keys, unexpected_keys = self.model.load_state_dict(
            filtered, strict=False
        )
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        log.info(
            "Loaded pretrained backbone from %s (matched %d/%d keys).",
            Path(checkpoint_path).expanduser(),
            len(filtered),
            len(model_state),
        )

        if strict:
            missing_backbone = [
                key
                for key in missing_keys
                if key.startswith((
                    "featurizer.", "decoder.", "backbone.", "avgpool.",
                    "vision_backbone.", "vision_proj.", "fusion_proj.", "head_vision.",
                    "temporal_attn.", "token_fusion.",
                ))
            ]
            if missing_backbone or unexpected_keys:
                raise RuntimeError(
                    "Error(s) in loading pretrained backbone:\n"
                    f"\tMissing backbone keys: {missing_backbone}\n"
                    f"\tUnexpected keys: {unexpected_keys}\n"
                )

    def _load_pretrained_angle_head(self, checkpoint_path: str, strict: bool = False) -> None:
        state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

        head = getattr(self.model, "head", None) or getattr(self.model, "angle_head", None)
        if head is None:
            log.warning("Model has neither 'head' nor 'angle_head' — skipping angle_head loading.")
            return

        head_state = head.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        matched = 0

        for key, value in state_dict.items():
            # Pretrain checkpoint uses "model.angle_head." prefix
            if key.startswith("model.angle_head."):
                mapped = key[len("model.angle_head.") :]
            elif key.startswith("angle_head."):
                mapped = key[len("angle_head.") :]
            # Regular model checkpoint uses "model.head." prefix
            elif key.startswith("model.head."):
                mapped = key[len("model.head.") :]
            elif key.startswith("head."):
                mapped = key[len("head.") :]
            else:
                continue

            if mapped not in head_state:
                continue

            target = head_state[mapped]
            if target.shape == value.shape:
                filtered[mapped] = value
                matched += 1
                continue

            if (
                value.ndim >= 1
                and target.shape[0] < value.shape[0]
                and target.shape[1:] == value.shape[1:]
            ):
                # Loaded has more dims than model: truncate
                filtered[mapped] = value[: target.shape[0]].clone()
                matched += 1
                continue

            if (
                value.ndim >= 1
                and target.shape[0] > value.shape[0]
                and target.shape[1:] == value.shape[1:]
            ):
                # Loaded has fewer dims than model: pad with zeros
                padded = torch.zeros_like(target)
                padded[: value.shape[0]] = value
                filtered[mapped] = padded
                matched += 1
                continue

        missing_keys, unexpected_keys = head.load_state_dict(
            filtered, strict=False
        )
        log.info(
            "Loaded pretrained angle_head from %s (matched %d/%d keys).",
            Path(checkpoint_path).expanduser(),
            matched,
            len(head_state),
        )
        if matched == 0:
            log.warning(
                "No angle_head weights matched. Check head architecture and "
                "out_channels vs pretrain."
            )

        if strict:
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Error(s) in loading pretrained angle_head:\n"
                    f"\tMissing keys: {missing_keys}\n"
                    f"\tUnexpected keys: {unexpected_keys}\n"
                )

    @rank_zero_only
    def _log_param_breakdown(self) -> None:
        def _count_params(module: torch.nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        model = self.model
        parts: dict[str, tuple[int, int]] = {}
        if getattr(model, "featurizer", None) is not None:
            parts["featurizer"] = _count_params(model.featurizer)
        if getattr(model, "decoder", None) is not None:
            parts["decoder"] = _count_params(model.decoder)

        if parts:
            formatted = ", ".join(
                f"{name}={total}/{trainable}"
                for name, (total, trainable) in parts.items()
            )
            log.info("Parameter breakdown (total/trainable): %s", formatted)

    def _freeze_backbone(self) -> None:
        for name, param in self.model.named_parameters():
            if name.startswith(("featurizer.", "decoder.")):
                param.requires_grad = False
        log.info("Backbone frozen (featurizer + decoder).")

    _COMPONENT_PREFIX_MAP: dict[str, list[str]] = {
        "featurizer": ["featurizer."],
        "decoder": ["decoder."],
        "vision_proj": ["vision_proj."],
        "fusion_proj": ["fusion_proj."],
        "head": ["head."],
        "head_vision": ["head_vision."],
        "vision_backbone": ["vision_backbone."],
        "backbone": ["backbone."],
    }

    def _params_by_component(self) -> dict[str, list[torch.nn.Parameter]]:
        """Group model parameters by component prefix."""
        groups: dict[str, list[torch.nn.Parameter]] = {c: [] for c in self._COMPONENT_PREFIX_MAP}
        unassigned: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            matched = False
            for comp, prefixes in self._COMPONENT_PREFIX_MAP.items():
                if any(name.startswith(p) for p in prefixes):
                    groups[comp].append(param)
                    matched = True
                    break
            if not matched:
                unassigned.append(param)
        if unassigned:
            groups["_unassigned"] = unassigned
        return groups

    def _freeze_component(self, component: str) -> None:
        prefixes = self._COMPONENT_PREFIX_MAP.get(component)
        if prefixes is None:
            log.warning("Unknown component '%s' for freezing, skipping.", component)
            return
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in prefixes):
                param.requires_grad = False
        log.info("Component '%s' frozen.", component)

    def _load_fusion_vision_weights(self, checkpoint_path: str) -> None:
        """Load vision_proj, fusion_proj, and head weights from a vision_only checkpoint."""
        state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

        vision_prefixes = ("vision_proj.", "head_vision.", "vision_backbone.")
        model_state = self.model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key[6:] if key.startswith("model.") else key
            if not any(stripped.startswith(p) for p in vision_prefixes):
                continue
            if stripped in model_state and model_state[stripped].shape == value.shape:
                filtered[stripped] = value

        if filtered:
            self.model.load_state_dict(filtered, strict=False)
            log.info(
                "Loaded %d vision/fusion/head keys from %s",
                len(filtered),
                Path(checkpoint_path).expanduser(),
            )
        else:
            log.warning("No matching vision/fusion/head keys found in %s", checkpoint_path)

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model.forward(batch)
        if self.task_type == "discrete":
            return self._prepare_discrete(out, batch)

        # Handle tuple output from BaseModule.forward() which returns (preds, targets, mask)
        if isinstance(out, tuple):
            preds, targets, mask = out
        # Handle dict output from EmgformerPretrain
        elif isinstance(out, dict):
            preds = out.get("angles", out.get("recon"))
            if preds is None:
                raise ValueError(
                    f"Model output dict must contain 'angles' or 'recon' key. "
                    f"Got keys: {list(out.keys())}."
                )
            # For dict output, derive targets and mask from batch
            joint_angles = batch.get("joint_angles", batch.get("angle_target"))
            mask = batch.get("label_valid_mask", batch.get("angle_mask"))
            if joint_angles is None or mask is None:
                raise KeyError(
                    "Batch must contain either (joint_angles, label_valid_mask) "
                    f"or (angle_target, angle_mask). Got keys: {list(batch.keys())}"
                )
            if joint_angles.shape[-1] == 1:
                # center-target-only: single center frame, extract center prediction
                targets = joint_angles
                center = preds.shape[-1] // 2
                preds = preds[..., center:center + 1]
            else:
                start = self.model.left_context
                stop = None if self.model.right_context == 0 else -self.model.right_context
                targets = joint_angles[..., slice(start, stop)]
                mask = mask[..., slice(start, stop)]
                # Align predictions and mask up to targets' time dimension
                n_time = targets.shape[-1]
                preds = self.model.align_predictions(preds, n_time)
                if mask.ndim == 2:
                    mask = self.model.align_mask(mask, n_time)
                elif mask.ndim == 3:
                    mask = self.model.align_mask(
                        mask.mean(dim=1), n_time
                    )  # (B, C, T) -> (B, T) -> align -> (B, T)
        else:
            # Legacy tensor-only output (e.g., Emg2PoseFormer, VEMG2PoseWithInitialState)
            preds = out
            joint_angles = batch.get("joint_angles", batch.get("angle_target"))
            mask = batch.get("label_valid_mask", batch.get("angle_mask"))
            if joint_angles is None or mask is None:
                raise KeyError(
                    "Batch must contain either (joint_angles, label_valid_mask) "
                    f"or (angle_target, angle_mask). Got keys: {list(batch.keys())}"
                )
            if joint_angles.shape[-1] == 1:
                # center-target-only: targets are single center frame,
                # extract center prediction step — no interpolation needed.
                targets = joint_angles
                center = preds.shape[-1] // 2
                preds = preds[..., center:center + 1]
            else:
                start = self.model.left_context
                stop = None if self.model.right_context == 0 else -self.model.right_context
                targets = joint_angles[..., slice(start, stop)]
                mask = mask[..., slice(start, stop)]
                # Align predictions and mask up to targets' time dimension
                n_time = targets.shape[-1]
                preds = self.model.align_predictions(preds, n_time)
                if mask.ndim == 2:
                    mask = self.model.align_mask(mask, n_time)
                elif mask.ndim == 3:
                    mask = self.model.align_mask(
                        mask.mean(dim=1), n_time
                    )  # (B, C, T) -> (B, T) -> align -> (B, T)

        if self.ignore_head_tail_dims > 0:
            if preds.ndim == 2:
                preds = preds[..., None]
            if self.ignore_head_tail_dims >= preds.shape[1]:
                raise ValueError(
                    "ignore_head_tail_dims must be smaller than prediction channels."
                )
            preds = preds[:, : -self.ignore_head_tail_dims, :]

        # Handle prediction-target channel mismatch (e.g., pretrain model outputs
        # 22 channels but dataset only provides 20 joint angles)
        if preds.shape[1] != targets.shape[1]:
            n_ch = min(preds.shape[1], targets.shape[1])
            preds = preds[:, :n_ch, :]
            targets = targets[:, :n_ch, :]

        # For 3D masks (angle_mask: B,C,T) the time is already aligned with targets;
        # for 2D masks (label_valid_mask: B,T) we need temporal alignment
        if mask.ndim == 3:
            # Already time-aligned; just trim channels if needed
            if mask.shape[1] != targets.shape[1]:
                mask = mask[:, :targets.shape[1], :]
        elif mask.ndim == 2:
            n_time = targets.shape[-1]
            mask = self.model.align_mask(mask, n_time)

        return preds, targets, mask

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str = "train"
    ) -> torch.Tensor:
        debug = _debug_steps_enabled()
        step_t0 = time.perf_counter()
        last_t = step_t0

        def mark(name: str) -> None:
            nonlocal last_t
            if not debug:
                return
            self._debug_sync()
            now = time.perf_counter()
            self._debug_step_log(
                f"{stage} batch_idx={getattr(self, '_debug_batch_idx', '?')} "
                f"{name}: +{now - last_t:.3f}s total={now - step_t0:.3f}s"
            )
            last_t = now

        if debug:
            emg = batch.get("emg")
            ja = batch.get("joint_angles", batch.get("angle_target"))
            mask = batch.get("label_valid_mask", batch.get("angle_mask"))
            self._debug_step_log(
                f"{stage} batch_idx={getattr(self, '_debug_batch_idx', '?')} "
                f"start emg={tuple(emg.shape) if emg is not None else None} "
                f"target={tuple(ja.shape) if ja is not None else None} "
                f"mask={tuple(mask.shape) if mask is not None else None}"
            )

        # Generate predictions
        if getattr(self.hparams, "datamodule", None) and self.hparams.datamodule.get("norm_mode") == "batch":
            emg = batch["emg"]
            mean = emg.mean()
            std = emg.std()
            batch["emg"] = (emg - mean) / (std + 1e-6)
            mark("batch_norm")
        preds, targets, mask = self.forward(batch)
        mark(
            f"forward preds={tuple(preds.shape)} targets={tuple(targets.shape)} "
            f"mask={tuple(mask.shape)}"
        )
        batch_size = batch["emg"].shape[0]

        # regression path
        valid_mask = mask.bool()
        mark("mask_bool")

        # Some datasets have no wrist annotations (channels 20,21 are
        # zero-padded). Mask wrist only for those samples, while keeping EgoEMG
        # wrist supervised in mixed batches.
        if "dataset_name" in batch:
            wrist_invalid_rows = torch.tensor(
                [d in {"egoemg_incre", "showee"} for d in batch["dataset_name"]],
                device=preds.device,
                dtype=torch.bool,
            )
            if wrist_invalid_rows.any() and preds.shape[1] > 20:
                if valid_mask.ndim == 2:
                    n_joints = preds.shape[1]
                    valid_mask = (
                        valid_mask.unsqueeze(1)
                        .expand(-1, n_joints, -1)
                        .clone()
                    )
                else:
                    valid_mask = valid_mask.clone()
                valid_mask[wrist_invalid_rows, -2:, :] = False  # wrist pitch/yaw
        mark("wrist_mask")

        metrics = {}
        for metric in self.regression_metrics:
            metric_t0 = time.perf_counter()
            metrics.update(metric(preds, targets, valid_mask, stage))
            if debug:
                self._debug_sync()
                self._debug_step_log(
                    f"{stage} batch_idx={getattr(self, '_debug_batch_idx', '?')} "
                    f"metric {metric.__class__.__name__}: "
                    f"{time.perf_counter() - metric_t0:.3f}s"
                )
        mark("metrics_all")
        self.log_dict(metrics, sync_dist=True, batch_size=batch_size)
        mark("log_dict")

        # Per-split MAE accumulation (val only, when generalization info available)
        if stage == "val" and "held_out_user" in batch and "held_out_stage" in batch:
            hou = batch["held_out_user"]   # (B,) BoolTensor
            hos = batch["held_out_stage"]  # (B,) BoolTensor
            per_elem_err = (preds - targets).abs()  # (B, J, T)
            # Expand valid_mask to match preds shape if needed
            if valid_mask.ndim == 2:
                vmask = valid_mask.unsqueeze(1).expand_as(per_elem_err)
            else:
                vmask = valid_mask
            # Per-bucket: stage=(!hou & hos), user=(hou & !hos), user_stage=(hou & hos)
            buckets = {
                "stage": (~hou) & hos,
                "user": hou & (~hos),
                "user_stage": hou & hos,
            }
            for bucket_name, bucket_mask in buckets.items():
                # bucket_mask: (B,) — expand to (B, 1, 1) to broadcast over (B, J, T)
                bmask = bucket_mask.view(-1, 1, 1) & vmask
                if bmask.any():
                    berr = (per_elem_err * bmask.float()).sum().item()
                    bcount = bmask.sum().item()
                    self._val_split_sums[bucket_name] = self._val_split_sums.get(bucket_name, 0.0) + berr
                    self._val_split_counts[bucket_name] = self._val_split_counts.get(bucket_name, 0.0) + bcount
            mark("val_split_accum")

        # Per-episode MAE accumulation (val only)
        if stage == "val" and "episode_id" in batch:
            ep_ids = batch["episode_id"]  # tuple of str, length B
            per_elem_err = (preds - targets).abs()  # (B, J, T)
            if valid_mask.ndim == 2:
                vmask_ep = valid_mask.unsqueeze(1).expand_as(per_elem_err)
            else:
                vmask_ep = valid_mask
            for i, ep_id in enumerate(ep_ids):
                e_err = per_elem_err[i]  # (J, T)
                e_mask = vmask_ep[i]    # (J, T)
                if e_mask.any():
                    self._val_episode_sums[ep_id] = (
                        self._val_episode_sums.get(ep_id, 0.0)
                        + (e_err * e_mask.float()).sum().item()
                    )
                    self._val_episode_counts[ep_id] = (
                        self._val_episode_counts.get(ep_id, 0.0)
                        + e_mask.sum().item()
                    )
            mark("val_episode_accum")

        # Per-joint MAE accumulation (val only)
        if stage == "val":
            per_elem_err = (preds - targets).abs()  # (B, J, T)
            if valid_mask.ndim == 2:
                vmask_j = valid_mask.unsqueeze(1).expand_as(per_elem_err)
            else:
                vmask_j = valid_mask
            for j in range(per_elem_err.shape[1]):
                j_err = per_elem_err[:, j, :]  # (B, T)
                j_mask = vmask_j[:, j, :]      # (B, T)
                if j_mask.any():
                    self._val_joint_sums[j] = (
                        self._val_joint_sums.get(j, 0.0)
                        + (j_err * j_mask.float()).sum().item()
                    )
                    self._val_joint_counts[j] = (
                        self._val_joint_counts.get(j, 0.0)
                        + j_mask.sum().item()
                    )
            mark("val_joint_accum")

        # ── Center-frame MAE (for vision / fusion models) ──────────
        # Only needed when T > 1 (broadcast case).  When the dataset already
        # returns center-frame-only targets (T == 1) the regular metrics above
        # cover the single time step.
        if ("vision_features" in batch or "vision_img" in batch) and mask.shape[-1] > 1:
            T = mask.shape[-1]
            center = T // 2
            center_mask = torch.zeros_like(mask, dtype=torch.float32)
            center_mask[..., center] = mask[..., center].float()
            center_valid = center_mask.bool()
            if center_valid.any():
                center_mae = {}
                for metric in self.regression_metrics:
                    center_mae.update(metric(preds, targets, center_valid, f"{stage}_center"))
                self.log_dict(center_mae, sync_dist=True, batch_size=batch_size)
                mark("center_metrics")

        # Backward-compat: map old flat loss-weight keys to new hierarchical metric keys.
        _LOSS_KEY_COMPAT = {
            "fingertip_distance": "landmark/fingertip",
            "landmark_distance": "landmark/all",
        }
        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            lookup = _LOSS_KEY_COMPAT.get(loss_name, loss_name)
            full_key = f"{stage}_{lookup}"
            metric_val = metrics.get(full_key, None)
            if metric_val is None:
                # Loss weight specified but no matching metric computed — usually
                # a typo in loss_weights config or an unsupported loss for this
                # model. Warn once (non-fatal: weight may legitimately be 0).
                if weight != 0.0 and not getattr(self, f"_warned_missing_{loss_name}", False):
                    log.warning(
                        "loss_weights entry %r (weight=%s) produced no metric "
                        "(key %r missing); this loss term is silently ignored. "
                        "Check for typos or unsupported loss for this model.",
                        loss_name, weight, full_key,
                    )
                    setattr(self, f"_warned_missing_{loss_name}", True)
                metric_val = 0.0
            loss += metric_val * weight
        mark("loss_reduce")

        # ── Delta L2: always report magnitude, optionally regularize ──
        delta_reg_weight = float(self.loss_weights.get("delta_reg", 0.0))
        delta = getattr(self.model, "_last_delta", None)
        if delta is not None:
            delta_l2 = (delta ** 2).mean()
            self.log(f"{stage}_delta_l2", delta_l2, sync_dist=True, batch_size=batch_size)
            if delta_reg_weight > 0 and stage == "train":
                loss = loss + delta_reg_weight * delta_l2

        # ── Invalid-EMG anchor: push the residual branch to 0 when EMG is
        # absent or deliberately mismatched, so delta cannot encode a visual
        # shortcut merely gated by the presence of a nonzero EMG signal.
        if (
            stage == "train"
            and self.anchor_loss_weight > 0.0
            and hasattr(self.model, "compute_anchor_emg_delta")
        ):
            anchor_delta = self.model.compute_anchor_emg_delta(
                batch, shuffle_fraction=self.anchor_shuffle_fraction
            )
            if anchor_delta is not None:
                anchor_l2 = (anchor_delta ** 2).mean()
                self.log(
                    f"{stage}_anchor_l2",
                    anchor_l2,
                    sync_dist=True,
                    batch_size=batch_size,
                )
                loss = loss + self.anchor_loss_weight * anchor_l2

        self.log(f"{stage}_loss", loss, sync_dist=True, batch_size=batch_size)
        mark("log_loss_return")
        return loss
        
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if _debug_steps_enabled():
            rank = getattr(self.trainer, "global_rank", 0) if self.trainer is not None else 0
            emg = batch.get("emg") if isinstance(batch, Mapping) else None
            print(
                f"[emg2pose-debug][rank={rank}] after_batch_transfer start "
                f"training={self.trainer.training if self.trainer is not None else None} "
                f"dataloader_idx={dataloader_idx} "
                f"emg={tuple(emg.shape) if emg is not None else None}",
                flush=True,
            )
            t0 = time.perf_counter()
        if self.batch_aug is not None and self.trainer.training:
            batch = self.batch_aug(batch)
        if _debug_steps_enabled():
            self._debug_sync()
            rank = getattr(self.trainer, "global_rank", 0) if self.trainer is not None else 0
            print(
                f"[emg2pose-debug][rank={rank}] after_batch_transfer done "
                f"{time.perf_counter() - t0:.3f}s",
                flush=True,
            )
        return batch

    def on_train_epoch_start(self) -> None:
        self._debug_step_log(f"train_epoch_start epoch={self.current_epoch}")

    def on_train_batch_start(self, batch, batch_idx) -> None:
        if not _debug_steps_enabled():
            return
        emg = batch.get("emg") if isinstance(batch, Mapping) else None
        self._debug_step_log(
            f"train_batch_start batch_idx={batch_idx} "
            f"emg={tuple(emg.shape) if emg is not None else None}"
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not _debug_steps_enabled():
            return batch
        rank = getattr(self.trainer, "global_rank", 0) if self.trainer is not None else 0
        emg = batch.get("emg") if isinstance(batch, Mapping) else None
        print(
            f"[emg2pose-debug][rank={rank}] before_batch_transfer "
            f"training={self.trainer.training if self.trainer is not None else None} "
            f"dataloader_idx={dataloader_idx} "
            f"emg={tuple(emg.shape) if emg is not None else None}",
            flush=True,
        )
        return batch

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self._debug_batch_idx = batch_idx
        result = self._step(batch, stage="train")
        # Log learning rate
        sch = self.lr_schedulers()
        if sch is not None:
            self.log("lr", sch.get_last_lr()[0], on_step=False, on_epoch=True, prog_bar=True)
        return result

    def on_validation_epoch_start(self) -> None:
        """Reset per-split and per-episode validation MAE accumulators."""
        self._val_split_sums = {"stage": 0.0, "user": 0.0, "user_stage": 0.0}
        self._val_split_counts = {"stage": 0.0, "user": 0.0, "user_stage": 0.0}
        self._val_episode_sums = {}
        self._val_episode_counts = {}
        self._val_joint_sums = {}
        self._val_joint_counts = {}

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        self._debug_batch_idx = batch_idx
        return self._step(batch, stage="val")

    def on_validation_epoch_end(self) -> None:
        """Compute and log per-split and per-episode validation MAE with DDP sync."""
        for bucket in ("stage", "user", "user_stage"):
            s = self._val_split_sums.get(bucket, 0.0)
            c = self._val_split_counts.get(bucket, 0.0)
            # DDP: gather sums and counts across all devices
            t_sum = torch.tensor(s, device=self.device, dtype=torch.float64)
            t_cnt = torch.tensor(c, device=self.device, dtype=torch.float64)
            gathered_sums = self.all_gather(t_sum)
            gathered_cnts = self.all_gather(t_cnt)
            total_sum = gathered_sums.sum()
            total_cnt = gathered_cnts.sum()
            if total_cnt > 0:
                mae_val = (total_sum / total_cnt).item()
                self.log(f"val_{bucket}_mae", mae_val, sync_dist=False,
                         batch_size=int(total_cnt.item()))

        # Per-episode validation MAE requires every rank to call collectives in
        # the exact same order. If no complete episode list is provided, each
        # DDP rank may see a different local set and deadlock here.
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        if self.val_episode_name_mapping or world_size <= 1:
            all_ep_ids = (
                sorted(self.val_episode_name_mapping.keys())
                if self.val_episode_name_mapping
                else sorted(self._val_episode_sums.keys())
            )
            for ep_id in all_ep_ids:
                s = self._val_episode_sums.get(ep_id, 0.0)
                c = self._val_episode_counts.get(ep_id, 0.0)
                t_sum = torch.tensor(s, device=self.device, dtype=torch.float64)
                t_cnt = torch.tensor(c, device=self.device, dtype=torch.float64)
                gathered_sums = self.all_gather(t_sum)
                gathered_cnts = self.all_gather(t_cnt)
                total_sum = gathered_sums.sum()
                total_cnt = gathered_cnts.sum()
                if total_cnt > 0:
                    mae_val = (total_sum / total_cnt).item()
                    display_name = self.val_episode_name_mapping.get(ep_id, ep_id)
                    self.log(
                        f"val_mae/{display_name}",
                        mae_val,
                        sync_dist=False,
                        batch_size=int(total_cnt.item()),
                    )

        # Per-joint validation MAE with DDP sync
        from egoemg.constants import JOINTS
        idx_to_name = {j.index: j.name for j in JOINTS}
        for j_idx in range(len(JOINTS)):
            s = self._val_joint_sums.get(j_idx, 0.0)
            c = self._val_joint_counts.get(j_idx, 0.0)
            t_sum = torch.tensor(s, device=self.device, dtype=torch.float64)
            t_cnt = torch.tensor(c, device=self.device, dtype=torch.float64)
            gathered_sums = self.all_gather(t_sum)
            gathered_cnts = self.all_gather(t_cnt)
            total_sum = gathered_sums.sum()
            total_cnt = gathered_cnts.sum()
            if total_cnt > 0:
                mae_val = (total_sum / total_cnt).item()
                joint_name = idx_to_name.get(j_idx, f"joint_{j_idx}")
                self.log(f"val_mae_per_joint/{joint_name}", mae_val, sync_dist=False,
                         batch_size=int(total_cnt.item()))

    def test_step(
        self, batch, batch_idx, dataloader_idx: int | None = None
    ) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        scales = self.component_lr_scales
        if not scales:
            return utils.instantiate_optimizer_and_scheduler(
                self.parameters(),
                optimizer_config=self.hparams.optimizer_conf,
                lr_scheduler_config=self.hparams.lr_scheduler_conf,
            )

        # Per-component param groups with scaled learning rates
        base_lr = float(self.hparams.optimizer_conf.lr)
        comp_params = self._params_by_component()
        param_groups = []
        for comp, params in comp_params.items():
            params = [p for p in params if p.requires_grad]
            if not params:
                continue
            scale = scales.get(comp, 1.0)
            param_groups.append({
                "params": params,
                "lr": base_lr * scale,
                "name": comp,
            })

        if not param_groups:
            raise RuntimeError("All parameters frozen — nothing to optimize.")

        names_and_scales = ", ".join(
            f"{g['name']}={scales.get(g['name'], 1.0):.0e}"
            for g in param_groups
        )
        log.info("Per-component LR scales: %s (base_lr=%.0e)", names_and_scales, base_lr)

        lr_scheduler_conf = self.hparams.lr_scheduler_conf
        return utils.instantiate_optimizer_and_scheduler(
            param_groups,
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=lr_scheduler_conf,
        )
