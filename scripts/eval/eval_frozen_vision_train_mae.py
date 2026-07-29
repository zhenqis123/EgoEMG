#!/usr/bin/env python
"""Measure the frozen vision branch on a fusion run's exact train dataset."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from emg2pose.datamodule import make_data_module
from emg2pose.train import make_lightning_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    cfg = OmegaConf.load(args.config)
    cfg.pretrained_checkpoint = None
    cfg.pretrained_emg_checkpoint = None
    cfg.stage2_vision_checkpoint = None
    OmegaConf.resolve(cfg)

    datamodule = make_data_module(cfg)
    datamodule.setup("fit")
    dataset = datamodule.train_dataset
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.num_workers) > 0,
        collate_fn=datamodule._collate_fn,
    )

    template = make_lightning_module(cfg)
    kwargs = {
        "module_conf": cfg.module,
        "optimizer_conf": cfg.optimizer,
        "lr_scheduler_conf": cfg.lr_scheduler,
        "loss_weights": cfg.loss_weights,
        "datamodule": cfg.datamodule,
        "pretrained_checkpoint": None,
        "pretrained_emg_checkpoint": None,
        "stage2_vision_checkpoint": None,
        "map_location": "cpu",
    }
    if "weights_only" in inspect.signature(
        template.__class__.load_from_checkpoint
    ).parameters:
        kwargs["weights_only"] = False
    lightning_module = template.__class__.load_from_checkpoint(
        args.checkpoint, **kwargs
    )
    model = lightning_module.model.to(device).eval()
    model.fusion_mode = "vision_only"

    error_sum = torch.zeros((), device=device, dtype=torch.float64)
    valid_count = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            preds, targets, mask = model(batch)
            valid = mask.bool()
            if valid.ndim == 2:
                valid = valid.unsqueeze(1).expand_as(preds)
            elif valid.shape != preds.shape:
                valid = valid.expand_as(preds)
            error_sum += (preds - targets).abs()[valid].double().sum()
            valid_count += valid.sum().double()
            sample_count += preds.shape[0]
            if rank == 0 and (batch_idx + 1) % 25 == 0:
                print(f"{batch_idx + 1}/{len(loader)}", flush=True)

    totals = torch.stack((error_sum, valid_count, sample_count))
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if rank == 0:
        result = {
            "train_mae": (totals[0] / totals[1]).item(),
            "valid_elements": int(totals[1].item()),
            "sample_count_with_sampler_padding": int(totals[2].item()),
            "dataset_length": len(dataset),
            "world_size": world_size,
            "batch_size_per_gpu": int(cfg.batch_size),
            "fusion_mode": "vision_only",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
