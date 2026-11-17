"""Exact user-macro MAE, deduplicating distributed evaluation padding."""

from collections import defaultdict

import torch


class UserMacroMAE:
    def __init__(self):
        self.windows = {}

    def update(self, batch, predictions, targets, mask):
        if mask.ndim == 2:
            mask = mask[:, None].expand_as(predictions)
        errors = (predictions - targets).abs().masked_fill(~mask.bool(), 0)
        sums = errors.double().sum((1, 2)).cpu().tolist()
        counts = mask.sum((1, 2)).cpu().tolist()
        # Lightning transfers metadata tensors to CUDA too. Copy them once per
        # batch rather than synchronizing the GPU for every window's scalar ID.
        sessions = batch['session_idx']
        starts = batch['window_start_idx']
        if isinstance(sessions, torch.Tensor):
            sessions = sessions.cpu().tolist()
        if isinstance(starts, torch.Tensor):
            starts = starts.cpu().tolist()
        for i, (error, count) in enumerate(zip(sums, counts)):
            key = (str(batch['user'][i]), int(sessions[i]), int(starts[i]))
            self.windows[key] = (error, count)

    @staticmethod
    def aggregate(shards):
        # DistributedSampler may repeat the final windows to equalize ranks.
        windows = {}
        for shard in shards:
            windows.update(shard)
        users = defaultdict(lambda: [0.0, 0])
        for (user, _, _), (error, count) in windows.items():
            users[user][0] += error
            users[user][1] += count
        means = [error/count for error, count in users.values() if count]
        if not means:
            raise ValueError('User-macro validation has no valid labels')
        return sum(means)/len(means)

    def compute(self):
        shards = [self.windows]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            shards = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(shards, self.windows)
        return self.aggregate(shards)
