from __future__ import annotations

import logging
from collections.abc import Mapping

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

from emg2pose.models.modules.cldm_vae import VAE1D, kl_divergence
from emg2pose.models.modules.cldm_unet import UNet1D
from emg2pose.utils import instantiate_optimizer_and_scheduler

log = logging.getLogger(__name__)


class LatentVAE1DModule(pl.LightningModule):
    def __init__(
        self,
        vae_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig | None,
        input_key: str,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: VAE1D = instantiate(vae_conf, _convert_="all")
        self.input_key = str(input_key)
        self.kl_weight = float(kl_weight)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch[self.input_key]
        if x.ndim != 3:
            raise ValueError(
                f"Expected input (B,C,T) for {self.input_key}, got {tuple(x.shape)}"
            )
        recon, mu, logvar = self.model(x)
        recon_loss = F.mse_loss(recon, x)
        kl = kl_divergence(mu, logvar)
        loss = recon_loss + self.kl_weight * kl
        self.log(f"{stage}/loss", loss, sync_dist=True)
        self.log(f"{stage}/recon_loss", recon_loss, sync_dist=True)
        self.log(f"{stage}/kl_loss", kl, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )


class CLDMDiffusionModule(pl.LightningModule):
    def __init__(
        self,
        unet_conf: DictConfig,
        emg_vae_conf: DictConfig,
        pose_vae_conf: DictConfig,
        emg_vae_checkpoint: str,
        pose_vae_checkpoint: str,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig | None,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        use_vae_mu: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["unet_conf", "emg_vae_conf", "pose_vae_conf"])
        self.unet: UNet1D = instantiate(unet_conf, _convert_="all")

        self.emg_vae = self._load_vae(emg_vae_conf, emg_vae_checkpoint)
        self.pose_vae = self._load_vae(pose_vae_conf, pose_vae_checkpoint)
        self._freeze_vae(self.emg_vae)
        self._freeze_vae(self.pose_vae)

        self.num_timesteps = int(num_timesteps)
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_cumprod", alpha_cumprod, persistent=False)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt(1.0 - alpha_cumprod),
            persistent=False,
        )
        self.use_vae_mu = bool(use_vae_mu)

    def _load_vae(self, conf: DictConfig, checkpoint_path: str) -> VAE1D:
        model: VAE1D = instantiate(conf, _convert_="all")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_state = {
            k.replace("model.", ""): v
            for k, v in state.items()
            if k.startswith("model.")
        }
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            log.warning("Missing VAE keys: %s", missing)
        if unexpected:
            log.warning("Unexpected VAE keys: %s", unexpected)
        return model

    def _freeze_vae(self, model: nn.Module) -> None:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    def _encode(self, model: VAE1D, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = model.encode(x)
        if self.use_vae_mu:
            return mu
        return model.reparameterize(mu, logvar)

    def _q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        return sqrt_alpha * z0 + sqrt_one_minus * noise

    def forward(self, emg: torch.Tensor, t: torch.Tensor, z_noisy: torch.Tensor) -> torch.Tensor:
        cond = self._encode(self.emg_vae, emg)
        return self.unet(z_noisy, t, cond)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        emg = batch["emg"]
        pose = batch["joint_angles"]
        if emg.ndim != 3 or pose.ndim != 3:
            raise ValueError(
                f"Expected emg/pose in (B,C,T), got {tuple(emg.shape)} and {tuple(pose.shape)}"
            )
        z_pose = self._encode(self.pose_vae, pose)
        b = z_pose.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=z_pose.device)
        noise = torch.randn_like(z_pose)
        z_noisy = self._q_sample(z_pose, t, noise)
        pred_noise = self.forward(emg, t, z_noisy)
        loss = F.mse_loss(pred_noise, noise)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        emg = batch["emg"]
        pose = batch["joint_angles"]
        z_pose = self._encode(self.pose_vae, pose)
        b = z_pose.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=z_pose.device)
        noise = torch.randn_like(z_pose)
        z_noisy = self._q_sample(z_pose, t, noise)
        pred_noise = self.forward(emg, t, z_noisy)
        loss = F.mse_loss(pred_noise, noise)
        self.log("val/loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return instantiate_optimizer_and_scheduler(
            self.unet.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )

    @torch.no_grad()
    def sample(self, emg: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        self.unet.eval()
        z_emg = self._encode(self.emg_vae, emg)
        num_steps = num_steps or self.num_timesteps
        z = torch.randn_like(z_emg)
        for step in reversed(range(num_steps)):
            t = torch.full((z.shape[0],), step, device=z.device, dtype=torch.long)
            pred_noise = self.unet(z, t, z_emg)
            alpha = self.alphas[t][:, None, None]
            alpha_bar = self.alpha_cumprod[t][:, None, None]
            beta = self.betas[t][:, None, None]
            if step > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z = (1.0 / torch.sqrt(alpha)) * (
                z - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise
        recon = self.pose_vae.decode(z)
        return recon
