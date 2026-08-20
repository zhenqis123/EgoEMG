"""GPU batch-level augmentation transforms for EMG data.

All transforms operate on batched tensors of shape (B, C, T) where:
  - B = batch size
  - C = EMG channels (16)
  - T = time steps (7790)

Randomness is per-sample via (B, ...) shaped tensors, generated directly on
the batch's device.  Expensive transforms use sub-batch extraction (gated
samples only) when probabilities are low; cheap transforms use full-batch
compute with gate masking.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def _per_sample_gate(
    batch_size: int, prob: float, device: torch.device
) -> torch.Tensor:
    """Return a boolean gate tensor of shape (B,)."""
    if prob <= 0.0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if prob >= 1.0:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return torch.rand(batch_size, device=device) < prob


def _linear_interp_1d(
    knot_x: torch.Tensor,  # (knots,)
    knot_y: torch.Tensor,  # (B, C, knots) or (G, C, knots)
    queries: torch.Tensor,  # (T,)
) -> torch.Tensor:
    """Vectorized linear interpolation for batched magnitude warping.

    Returns warping curves of shape (*, C, T).
    """
    idx = torch.searchsorted(knot_x, queries) - 1  # (T,)
    idx = idx.clamp(0, len(knot_x) - 2)

    x0 = knot_x[idx]  # (T,)
    x1 = knot_x[idx + 1]  # (T,)
    y0 = knot_y[..., idx]  # (*, C, T)
    y1 = knot_y[..., idx + 1]  # (*, C, T)

    denom = (x1 - x0).clamp(min=1e-8)  # (T,)
    frac = ((queries - x0) / denom).clamp(0.0, 1.0)  # (T,)
    return y0 + frac * (y1 - y0)


class BatchAugmentation(nn.Module):
    """GPU batch-level EMG augmentation.

    Applies a chain of augmentations to the "emg" tensor in a batch dict.
    All operations are vectorized across the batch dimension.

    Parameters are read from a Hydra-compatible config dict at init time
    and can be overridden per-trial via Optuna.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = config or {}

        # ── RandomGain ──────────────────────────────────────────────────
        gc = cfg.get("random_gain", {})
        self.gain_min: float = float(gc.get("min_gain", 0.8))
        self.gain_max: float = float(gc.get("max_gain", 1.25))
        self.gain_mask_prob: float = float(gc.get("mask_prob", 0.5))

        # ── RandomMagnitudeWarping ──────────────────────────────────────
        wc = cfg.get("mag_warping", {})
        self.warp_sigma: float = float(wc.get("sigma", 0.15))
        self.warp_num_knots: int = int(wc.get("num_knots", 8))
        self.warp_mask_prob: float = float(wc.get("mask_prob", 0.5))

        # ── BaselineDrift ───────────────────────────────────────────────
        dc = cfg.get("baseline_drift", {})
        self.drift_mask_prob: float = float(dc.get("mask_prob", 0.2))
        self.drift_min_freq: float = float(dc.get("min_freq", 0.05))
        self.drift_max_freq: float = float(dc.get("max_freq", 0.5))
        self.drift_min_amp: float = float(dc.get("min_amp_ratio", 0.02))
        self.drift_max_amp: float = float(dc.get("max_amp_ratio", 0.08))
        self.drift_sample_rate: float = 2000.0

        # ── PowerlineNoise ──────────────────────────────────────────────
        pc = cfg.get("powerline_noise", {})
        self.powerline_mask_prob: float = float(pc.get("mask_prob", 0.2))
        self.powerline_min_amp: float = float(pc.get("min_amp_ratio", 0.005))
        self.powerline_max_amp: float = float(pc.get("max_amp_ratio", 0.03))
        self.powerline_max_harmonic: int = int(pc.get("max_harmonic", 3))
        self.powerline_base_freq: float = 50.0  # fixed

        # ── RandomChannelMask ───────────────────────────────────────────
        cc = cfg.get("channel_mask", {})
        self.channel_mask_prob: float = float(cc.get("mask_prob", 0.036))
        self.channel_mask_value: float = float(cc.get("mask_value", 0.0))

        # ── RandomTimeMask ──────────────────────────────────────────────
        tc = cfg.get("time_mask", {})
        self.time_num_masks: int = int(tc.get("num_masks", 9))
        self.time_max_mask_size: int = int(tc.get("max_mask_size", 500))
        self.time_min_mask_size: int = int(tc.get("min_mask_size", 0))
        self.time_mask_value: float = float(tc.get("mask_value", 0.0))
        self.time_mask_per_channel: bool = bool(tc.get("per_channel", False))

        # ── RandomFrequencyMask ─────────────────────────────────────────
        fc = cfg.get("freq_mask", {})
        self.freq_num_masks: int = int(fc.get("num_masks", 3))
        self.freq_max_mask_size: int = int(fc.get("max_mask_size", 128))
        self.freq_min_mask_size: int = int(fc.get("min_mask_size", 0))

        # ── RandomGaussianNoise ─────────────────────────────────────────
        nc = cfg.get("gaussian_noise", {})
        self.noise_min_snr_db: float = float(nc.get("min_snr_db", 39.37))
        self.noise_max_snr_db: float = float(nc.get("max_snr_db", 50.0))
        self.noise_apply_prob: float = float(nc.get("apply_prob", 0.96))

        # ── ChannelRotation (electrode ring shift) ──────────────────────
        rc = cfg.get("channel_rotation", {})
        self.rotation_prob: float = float(rc.get("mask_prob", 0.0))
        self.rotation_max_shift: int = int(rc.get("max_shift", 2))

        # ── MixUp (frequency-domain inter-sample blending) ──────────────
        mc = cfg.get("mixup", {})
        self.mixup_prob: float = float(mc.get("mask_prob", 0.0))
        self.mixup_alpha: float = float(mc.get("alpha", 0.2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply all augmentations to batch["emg"] in-place."""
        emg = batch.get("emg")
        if emg is None or not self.training:
            return batch

        # Ensure (B, C, T) layout
        if emg.dim() != 3:
            return batch  # unexpected shape, skip

        # Apply transforms (each is a no-op when prob == 0)
        # Channel rotation runs first — models electrode shift, the physical
        # reality that precedes all other signal artifacts.
        emg = self._channel_rotation(emg)
        emg = self._random_gain(emg)
        emg = self._mag_warping(emg)
        emg = self._baseline_drift(emg)
        emg = self._powerline_noise(emg)
        emg = self._channel_mask(emg)
        emg = self._time_mask(emg)
        emg = self._freq_mask(emg)
        emg = self._gaussian_noise(emg)

        batch["emg"] = emg

        # MixUp runs last because it blends pairs across the batch dimension,
        # requiring joint_angles to be mixed too.
        batch = self._mixup(batch)

        return batch

    # ------------------------------------------------------------------
    # Individual transforms
    # ------------------------------------------------------------------

    def _random_gain(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample multiplicative gain (B, C, T)."""
        if self.gain_mask_prob <= 0.0:
            return x
        B, C = x.shape[0], x.shape[1]
        gate = _per_sample_gate(B, self.gain_mask_prob, x.device)
        if not gate.any():
            return x

        log_min = math.log(self.gain_min)
        log_max = math.log(self.gain_max)
        gains = torch.exp(
            torch.empty(B, C, device=x.device).uniform_(log_min, log_max)
        )  # (B, C)
        gains[~gate] = 1.0
        return (x.float() * gains.unsqueeze(2)).to(x.dtype)

    def _mag_warping(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample magnitude warping via linear interpolation (B, C, T)."""
        if self.warp_mask_prob <= 0.0 or self.warp_sigma <= 0.0:
            return x
        B, C, T = x.shape
        gate = _per_sample_gate(B, self.warp_mask_prob, x.device)
        if not gate.any():
            return x

        idx = gate.nonzero(as_tuple=True)[0]
        N = len(idx)

        knots = self.warp_num_knots
        knot_x = torch.linspace(0, T - 1, knots, device=x.device)
        knot_y = torch.normal(
            mean=1.0, std=self.warp_sigma,
            size=(N, C, knots), device=x.device, dtype=torch.float32,
        )

        queries = torch.arange(T, device=x.device, dtype=torch.float32)
        curves = _linear_interp_1d(knot_x, knot_y, queries)  # (N, C, T)

        x_f = x.float()
        x_f[idx] = x_f[idx] * curves
        return x_f.to(x.dtype)

    def _baseline_drift(self, x: torch.Tensor) -> torch.Tensor:
        """Batched baseline drift: sine + random walk — sub-batch."""
        if self.drift_mask_prob <= 0.0:
            return x
        B, C, T = x.shape
        gate = _per_sample_gate(B, self.drift_mask_prob, x.device)
        if not gate.any():
            return x

        idx = gate.nonzero(as_tuple=True)[0]
        N = len(idx)

        x_f = x.float()
        xs = x_f[idx]  # (N, C, T)
        t_arr = torch.arange(T, device=x.device, dtype=torch.float32) / self.drift_sample_rate
        scale = xs.std(dim=2, keepdim=True)

        freq = (
            torch.rand(N, device=x.device)
            * (self.drift_max_freq - self.drift_min_freq)
            + self.drift_min_freq
        )
        phase = torch.rand(N, C, 1, device=x.device) * (2.0 * math.pi)
        amp = (
            torch.rand(N, C, 1, device=x.device)
            * (self.drift_max_amp - self.drift_min_amp)
            + self.drift_min_amp
        )
        drift_sin = (scale * amp) * torch.sin(
            2.0 * math.pi * freq.view(N, 1, 1) * t_arr.view(1, 1, T) + phase
        )

        walk_std = 0.002 * scale
        steps = torch.randn(N, C, T, device=x.device, dtype=torch.float32) * walk_std
        walk = torch.cumsum(steps, dim=2)
        walk = walk - walk.mean(dim=2, keepdim=True)

        x_f[idx] = xs + drift_sin + walk
        return x_f.to(x.dtype)

    def _powerline_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Batched powerline noise with harmonics — sub-batch."""
        if self.powerline_mask_prob <= 0.0:
            return x
        B, C, T = x.shape
        gate = _per_sample_gate(B, self.powerline_mask_prob, x.device)
        if not gate.any():
            return x

        idx = gate.nonzero(as_tuple=True)[0]
        N = len(idx)

        x_f = x.float()
        xs = x_f[idx]
        t_arr = torch.arange(T, device=x.device, dtype=torch.float32) / self.drift_sample_rate
        scale = xs.std(dim=2, keepdim=True)

        noise = torch.zeros_like(xs)
        for harmonic in range(1, self.powerline_max_harmonic + 1):
            freq = self.powerline_base_freq * harmonic
            phase = torch.rand(N, C, 1, device=x.device) * (2.0 * math.pi)
            amp = (
                torch.rand(N, C, 1, device=x.device)
                * (self.powerline_max_amp - self.powerline_min_amp)
                + self.powerline_min_amp
            )
            noise = noise + (scale * amp) * torch.sin(
                2.0 * math.pi * freq * t_arr.view(1, 1, T) + phase
            )

        x_f[idx] = xs + noise
        return x_f.to(x.dtype)

    def _channel_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample channel masking (B, C, T)."""
        if self.channel_mask_prob <= 0.0:
            return x
        B, C = x.shape[0], x.shape[1]
        mask = torch.rand(B, C, device=x.device) < self.channel_mask_prob
        if not mask.any():
            return x
        x = x.clone()
        x[mask] = self.channel_mask_value
        return x

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample time masking (B, C, T) — vectorized, no per-sample loop.

        When ``time_mask_per_channel`` is True, each channel gets an independent
        mask position (modelling per-electrode signal dropout).  When False
        (legacy), the same mask is broadcast across all channels (modelling a
        full-sensor blackout).
        """
        if self.time_num_masks <= 0 or self.time_max_mask_size <= 0:
            return x
        B, C, T = x.shape
        min_sz = min(self.time_min_mask_size, self.time_max_mask_size)
        max_sz = max(self.time_min_mask_size, self.time_max_mask_size)

        if self.time_mask_per_channel:
            # Per-channel independent masks: shape (B, C) for size and start.
            mask_shape = (B, C)
        else:
            # Legacy: same mask across all channels.
            # Keep the singleton channel axis so unsqueeze(-1) produces
            # (B, 1, 1), which broadcasts correctly against (B, C, T).
            mask_shape = (B, 1)

        time_range = torch.arange(T, device=x.device).view(1, 1, T)

        for _ in range(self.time_num_masks):
            if min_sz == max_sz:
                mask_size = torch.full(
                    mask_shape, min(max_sz, T),
                    device=x.device, dtype=torch.long,
                )
            else:
                mask_size = torch.randint(
                    min_sz, max_sz + 1,
                    mask_shape, device=x.device,
                ).clamp(max=T)

            max_start = (T - mask_size).clamp(min=0)
            start = (
                torch.rand(mask_shape, device=x.device) * (max_start.float() + 1)
            ).long()
            start = start.clamp(max=T - 1)

            # Build mask of shape expand_dims (either (B,C,T) or (B,1,T))
            t_mask = (
                (time_range >= start.unsqueeze(-1))
                & (time_range < (start + mask_size).unsqueeze(-1))
            )
            x = x.masked_fill(t_mask, self.time_mask_value)
        return x

    def _freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample frequency-domain masking via batched rFFT — vectorized."""
        if self.freq_num_masks <= 0 or self.freq_max_mask_size <= 0:
            return x

        T = x.shape[2]
        B, C = x.shape[0], x.shape[1]
        spec = torch.fft.rfft(x.float(), dim=2)  # (B, C, n_freq)
        n_freq = spec.shape[2]
        min_sz = min(self.freq_min_mask_size, self.freq_max_mask_size)
        max_sz = max(self.freq_min_mask_size, self.freq_max_mask_size)
        freq_range = torch.arange(n_freq, device=x.device).view(1, 1, n_freq)

        for _ in range(self.freq_num_masks):
            if min_sz == max_sz:
                mask_size = torch.full(
                    (B,), min(max_sz, n_freq),
                    device=x.device, dtype=torch.long,
                )
            else:
                mask_size = torch.randint(
                    min_sz, max_sz + 1,
                    (B,), device=x.device,
                ).clamp(max=n_freq)

            max_start = (n_freq - mask_size).clamp(min=0)
            start = (
                torch.rand(B, device=x.device) * (max_start.float() + 1)
            ).long()
            start = start.clamp(max=n_freq - 1)

            f_mask = (
                (freq_range >= start.view(B, 1, 1))
                & (freq_range < (start + mask_size).view(B, 1, 1))
            )  # (B, 1, n_freq)
            spec.masked_fill_(f_mask.expand(B, C, n_freq), 0)

        return torch.fft.irfft(spec, n=T, dim=2).to(x.dtype)

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Batched additive Gaussian noise at specified SNR — sub-batch."""
        if self.noise_apply_prob <= 0.0:
            return x

        B = x.shape[0]
        gate = _per_sample_gate(B, self.noise_apply_prob, x.device)
        if not gate.any():
            return x

        x_f = x.float()
        signal_power = x_f.pow(2).mean(dim=(1, 2))  # (B,)
        valid = signal_power > 0
        apply = gate & valid
        if not apply.any():
            return x

        idx = apply.nonzero(as_tuple=True)[0]
        N = len(idx)

        xs = x_f[idx]  # (N, C, T)
        snr_db = (
            torch.rand(N, device=x.device)
            * (self.noise_max_snr_db - self.noise_min_snr_db)
            + self.noise_min_snr_db
        )
        snr_linear = 10.0 ** (snr_db / 10.0)

        noise_power = signal_power[idx] / snr_linear.clamp(min=1e-10)
        noise_std = noise_power.sqrt()

        noise = torch.randn_like(xs) * noise_std.view(N, 1, 1)
        x_f[idx] = xs + noise
        return x_f.to(x.dtype)

    def _channel_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample channel ring rotation (B, C, T) — simulates electrode shift.

        Rolls the channel dimension by a random integer in
        [-max_shift, +max_shift] for each gated sample, modelling the physical
        rotation of a wristband EMG sensor between sessions.
        """
        if self.rotation_prob <= 0.0 or self.rotation_max_shift <= 0:
            return x
        B, C, T = x.shape
        gate = _per_sample_gate(B, self.rotation_prob, x.device)
        if not gate.any():
            return x

        # Random shift per sample: 0 is a no-op even when gated.
        shifts = torch.randint(
            -self.rotation_max_shift, self.rotation_max_shift + 1,
            (B,), device=x.device,
        )
        shifts[~gate] = 0
        if not shifts.any():
            return x

        x_out = x.clone()
        for i in range(B):
            if shifts[i] != 0:
                x_out[i] = torch.roll(x[i], shifts=int(shifts[i]), dims=0)
        return x_out

    def _mixup(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Frequency-domain MixUp across the batch dimension.

        For each gated sample, blend its rFFT spectrum with a randomly chosen
        partner sample from the same batch, using a Beta(alpha, alpha) mixing
        coefficient.  The corresponding ``joint_angles`` target is blended with
        the same coefficient so the regression label stays consistent.
        """
        if self.mixup_prob <= 0.0 or self.mixup_alpha <= 0.0:
            return batch
        emg = batch.get("emg")
        if emg is None:
            return batch
        B, C, T = emg.shape
        gate = _per_sample_gate(B, self.mixup_prob, emg.device)
        if not gate.any():
            return batch

        # Partner indices: shuffle the batch to create random pairs.
        perm = torch.randperm(B, device=emg.device)

        # Mixing coefficient ~ Beta(alpha, alpha), per sample.
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample(
            (B,)
        ).to(emg.device).float()
        lam = torch.clamp(lam, 0.0, 1.0)
        # No mixing for un-gated samples (lam=1 → keep original).
        lam[~gate] = 1.0
        lam_exp = lam.view(B, 1, 1)

        # Mix in frequency domain for smoother blending.
        spec_orig = torch.fft.rfft(emg.float(), dim=2)
        spec_partner = torch.fft.rfft(emg[perm].float(), dim=2)
        spec_mixed = spec_orig * lam_exp + spec_partner * (1.0 - lam_exp)
        batch["emg"] = torch.fft.irfft(spec_mixed, n=T, dim=2).to(emg.dtype)

        # Blend the regression target if present.
        for key in ("joint_angles", "target", "labels"):
            tgt = batch.get(key)
            if tgt is None or not torch.is_tensor(tgt):
                continue
            tgt_f = tgt.float()
            lam_t = lam.view(B, *([1] * (tgt.dim() - 1)))
            batch[key] = (
                tgt_f * lam_t + tgt_f[perm] * (1.0 - lam_t)
            ).to(tgt.dtype)

        # Blend the validity mask with the same coefficient: when the partner
        # sample's target is an invalid placeholder (e.g. Incre left-hand zero
        # poses), the mixed target is only partially trustworthy, so the loss
        # mask must reflect the same mixture instead of keeping the gated
        # sample's original mask.
        mask = batch.get("label_valid_mask")
        if mask is not None and torch.is_tensor(mask):
            lam_m = lam.view(B, *([1] * (mask.dim() - 1)))
            mask_f = mask.to(torch.float32)
            mixed_mask = mask_f * lam_m + mask_f[perm] * (1.0 - lam_m)
            batch["label_valid_mask"] = (mixed_mask > 0.5).to(mask.dtype)

        return batch
