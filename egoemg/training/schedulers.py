"""Learning-rate schedules for explicitly scaled optimizer groups."""

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class ProportionalCosineAnnealingLR(LRScheduler):
    """Cosine with a common multiplicative factor for all initial group LRs.

    eta_min is the floor of optimizer.defaults['lr']; every group's own floor
    scales with its initial LR. Thus 1e-5/1e-4 groups end at 1e-6/1e-5 when
    the base optimizer LR is 1e-4 and eta_min=1e-5. State is fully serializable.
    """

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.reference_lr = float(optimizer.defaults["lr"])
        if not 0 <= self.eta_min <= self.reference_lr or self.reference_lr <= 0:
            raise ValueError("Require 0 <= eta_min <= positive reference LR")
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        floor = self.eta_min / self.reference_lr
        factor = (
            floor
            + (1 - floor)
            * (1 + math.cos(math.pi * min(self.last_epoch, self.T_max) / self.T_max))
            / 2
        )
        return [lr * factor for lr in self.base_lrs]


class WarmupCosineLR(LRScheduler):
    """Cosine schedule with optional fixed-length update periods.

    The default period length of one preserves epoch-driven behavior. With
    ``interval=step`` and ``steps_per_period=N``, schedule progress advances
    every N optimizer updates, independently of training epoch boundaries.
    """

    def __init__(
        self,
        optimizer,
        T_max,
        warmup_epochs=2,
        eta_min=0.0,
        start_factor=0.1,
        last_epoch=-1,
        steps_per_period=1,
    ):
        if not 0 <= warmup_epochs < T_max or not 0 < start_factor <= 1:
            raise ValueError("Invalid warmup schedule")
        if not isinstance(steps_per_period, int) or steps_per_period < 1:
            raise ValueError("steps_per_period must be a positive integer")
        self.steps_per_period = steps_per_period
        self.T_max = int(T_max)
        self.warmup_epochs = int(warmup_epochs)
        self.start_factor = float(start_factor)
        self.floor_factor = float(eta_min) / float(optimizer.defaults["lr"])
        if not 0 <= self.floor_factor <= 1:
            raise ValueError("Require 0 <= eta_min <= base optimizer LR")
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        period = self.last_epoch // self.steps_per_period
        if period < self.warmup_epochs:
            factor = (
                self.start_factor
                + (1 - self.start_factor) * period / self.warmup_epochs
            )
        else:
            progress = min(
                1.0,
                (period - self.warmup_epochs)
                / (self.T_max - self.warmup_epochs),
            )
            factor = (
                self.floor_factor
                + (1 - self.floor_factor) * (1 + math.cos(math.pi * progress)) / 2
            )
        return [lr * factor for lr in self.base_lrs]


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    max_epochs: int,
    eta_min: float = 1e-5,
    start_factor: float = 1e-3,
    end_factor: float = 1.0,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Helper to build Linear warmup + CosineAnnealing scheduler."""
    start_factor = max(start_factor, 1e-6)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=warmup_iters,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters]
    )
