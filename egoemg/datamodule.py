from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from egoemg import transforms
from egoemg.datasets.pretrain_wrapper import _resolve_transform

log = logging.getLogger(__name__)


class SessionShuffleSampler(DistributedSampler):
    """Session-grouped sampler for memmap-backed window datasets.

    Window reads from a multi-hundred-GB memmap are only fast when they are
    sequential. This sampler keeps each rank's reads session-sequential while
    approximating uniform shuffling:

    1. Sessions are shuffled and their index streams concatenated in temporal
       order, then split into ``num_replicas`` equal contiguous chunks
       (cyclically padded, exactly like ``DistributedSampler``): perfect
       balance, exact coverage, deterministic on every rank, no communication.
    2. The rank's chunk is cut into blocks of ``batch_size * block_batches``
       consecutive windows. Block order is shuffled across the chunk and
       window order is shuffled within each block. One block's span stays
       resident in the page cache, so intra-block random access is RAM-cheap
       while disk still only sees block-sized sequential reads.

    Subclassing ``DistributedSampler`` makes Lightning treat instances as
    already distributed-aware (no re-wrapping) and lets it call
    ``set_epoch()`` so sessions and blocks are re-shuffled every epoch.
    """

    def __init__(
        self,
        session_groups: Sequence[np.ndarray],
        *,
        batch_size: int,
        block_batches: int = 16,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid rank {rank} for {num_replicas} replicas")

        self.session_groups = [np.asarray(g, dtype=np.int64) for g in session_groups]
        self.batch_size = int(batch_size)
        self.block_batches = int(block_batches)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        total = sum(len(g) for g in self.session_groups)
        # Pad each rank's shard to a whole number of blocks (cyclic
        # repetition, like DistributedSampler padding): every yielded block
        # is exactly block_size, the shard divides evenly into batches, and
        # ~2% of windows repeat per epoch.
        block = max(1, self.batch_size * self.block_batches)
        per_rank = math.ceil(total / self.num_replicas) if total else 0
        self.num_samples = math.ceil(per_rank / block) * block
        self.total_size = self.num_samples * self.num_replicas

    def _rank_stream(self, epoch: int) -> np.ndarray:
        if not self.session_groups:
            return np.empty(0, dtype=np.int64)
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(len(self.session_groups))
        stream = np.concatenate([self.session_groups[si] for si in order])
        if len(stream) < self.total_size:  # cyclic padding, like DistributedSampler
            reps = -(-self.total_size // len(stream))
            stream = np.tile(stream, reps)[: self.total_size]
        return stream[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        stream = self._rank_stream(self.epoch)
        if len(stream) == 0:
            return iter(())
        block_size = max(1, self.batch_size * self.block_batches)
        blocks = [stream[i : i + block_size] for i in range(0, len(stream), block_size)]
        rng.shuffle(blocks)  # blocks is a list of ndarrays
        for b in blocks:
            rng.shuffle(b)
        return iter(int(i) for b in blocks for i in b)

    def __len__(self) -> int:
        return self.num_samples


def _debug_steps_enabled() -> bool:
    return os.environ.get("EMG2POSE_DEBUG_STEPS", "0").lower() in {"1", "true", "yes"}


def make_data_module(config: DictConfig) -> WindowedEmgDataModule:
    """Create and configure datamodule from experiment config."""
    dataset_conf = OmegaConf.to_container(config.dataset, resolve=True)
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_conf=dataset_conf,
        _recursive_=False,
    )
    datamodule.train_transforms = transforms.Compose(
        [instantiate(cfg) for cfg in config.transforms.train]
    )
    datamodule.val_transforms = transforms.Compose(
        [instantiate(cfg) for cfg in config.transforms.val]
    )
    datamodule.test_transforms = transforms.Compose(
        [instantiate(cfg) for cfg in config.transforms.test]
    )
    return datamodule


class _EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Any:
        raise IndexError(idx)


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        stride: int | None,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        eval_num_workers: int | None = None,
        multiprocessing_context: str | None = None,
        eval_multiprocessing_context: str | None = None,
        val_test_window_length: int | None = None,
        val_test_stride: int | None = None,
        eval_center_stride: int | None = None,
        skip_ik_failures: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        max_open_files: int = 32,
        session_shuffle: bool = True,
        session_block_batches: int = 16,
        dataset_repeat: int = 1,
        norm_mode: str | None = None,
        norm_stats_path: str | None = None,
        per_dataset_norm_stats_path: str | None = None,
        norm_eps: float = 1e-6,
        dataset_conf: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.stride = stride
        self.val_test_stride = val_test_stride if val_test_stride is not None else stride
        self.eval_center_stride = eval_center_stride
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_num_workers = num_workers if eval_num_workers is None else eval_num_workers
        self.multiprocessing_context = multiprocessing_context
        self.eval_multiprocessing_context = (
            multiprocessing_context
            if eval_multiprocessing_context is None
            else eval_multiprocessing_context
        )
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.max_open_files = max_open_files
        self.session_shuffle = session_shuffle
        self.session_block_batches = session_block_batches
        self.dataset_repeat = dataset_repeat
        self.norm_mode = norm_mode
        self.norm_stats_path = norm_stats_path
        self.per_dataset_norm_stats_path = per_dataset_norm_stats_path
        self.norm_eps = norm_eps

        self.dataset_conf = dataset_conf or {}
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _normalize_dataset_configs(self, cfgs: Any) -> list[Any]:
        if cfgs is None:
            return []
        if OmegaConf.is_dict(cfgs):
            return [cfgs]
        if OmegaConf.is_list(cfgs):
            return list(cfgs)
        if isinstance(cfgs, Sequence) and not isinstance(cfgs, (str, bytes)):
            return list(cfgs)
        return [cfgs]

    def _build_dataset(self, split: str, transform: Any) -> Dataset:
        cfgs = self._normalize_dataset_configs(self.dataset_conf.get(split, []))
        datasets: list[Dataset] = []
        dataset_info: list[tuple[str, int]] = []

        for cfg in cfgs:
            # Inject norm settings into PretrainWrapperDataset configs
            if self.norm_mode == "per-dataset":
                cfg = OmegaConf.merge(
                    cfg,
                    {
                        "norm_mode": self.norm_mode,
                        "norm_stats_path": self.per_dataset_norm_stats_path,
                    },
                )

            # Instantiate without transform parameter
            dataset = instantiate(cfg, transform=None)
            # Set transform directly after instantiation to avoid conversion
            if hasattr(dataset, 'transform'):
                dataset.transform = transform
            # Also set _transform if it exists (for PretrainWrapperDataset)
            if hasattr(dataset, '_transform'):
                dataset._transform = _resolve_transform(transform) if transform is not None else None

            # Collect dataset info
            dataset_name = getattr(dataset, 'name', dataset.__class__.__name__)
            dataset_len = len(dataset)
            dataset_info.append((dataset_name, dataset_len))
            datasets.append(dataset)

        if not datasets:
            return _EmptyDataset()

        # Print dataset composition with aggregation
        if dataset_info:
            total_samples = sum(length for _, length in dataset_info)

            # Aggregate by dataset name
            from collections import defaultdict
            aggregated = defaultdict(lambda: {'count': 0, 'samples': 0})
            for name, length in dataset_info:
                aggregated[name]['count'] += 1
                aggregated[name]['samples'] += length

            log.info(f"\n{'='*70}")
            log.info(f"Dataset composition for '{split}' split:")
            log.info(f"{'-'*70}")

            for name in sorted(aggregated.keys()):
                info = aggregated[name]
                count = info['count']
                samples = info['samples']
                percentage = (samples / total_samples * 100) if total_samples > 0 else 0

                if count > 1:
                    log.info(f"  {name:20s}: {samples:8d} samples ({percentage:5.2f}%) [{count} subsets]")
                else:
                    log.info(f"  {name:20s}: {samples:8d} samples ({percentage:5.2f}%)")

            log.info(f"{'-'*70}")
            log.info(f"  {'Total':20s}: {total_samples:8d} samples")
            log.info(f"{'='*70}\n")

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)

        if self.dataset_repeat > 1:
            dataset = ConcatDataset([dataset] * self.dataset_repeat)
            log.info("Dataset repeated %dx → %d total samples", self.dataset_repeat, len(dataset))
        return dataset

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train", self.train_transforms)
            self.val_dataset = self._build_dataset("val", self.val_transforms)
        if stage in (None, "validate"):
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset("val", self.val_transforms)
        if stage in (None, "test"):
            self.test_dataset = self._build_dataset("test", self.test_transforms)
        if stage in (None, "predict"):
            if self.test_dataset is None:
                self.test_dataset = self._build_dataset("test", self.test_transforms)

    def _make_loader(
        self,
        dataset: Dataset,
        shuffle: bool,
        *,
        num_workers: int | None = None,
        multiprocessing_context: str | None = None,
    ) -> DataLoader:
        loader_num_workers = self.num_workers if num_workers is None else num_workers
        loader_multiprocessing_context = (
            self.multiprocessing_context
            if multiprocessing_context is None
            else multiprocessing_context
        )
        if _debug_steps_enabled():
            rank = os.environ.get("RANK", "0")
            print(
                f"[emg2pose-debug][rank={rank}] make_loader "
                f"shuffle={shuffle} len={len(dataset)} batch_size={self.batch_size} "
                f"num_workers={loader_num_workers} persistent_workers={self.persistent_workers} "
                f"prefetch_factor={self.prefetch_factor if loader_num_workers > 0 else None} "
                f"multiprocessing_context={loader_multiprocessing_context}",
                flush=True,
            )
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": loader_num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": self._collate_fn,  # Use custom collate function
        }
        worker_init = getattr(dataset, "_worker_init", None)
        if worker_init is not None and loader_num_workers > 0:
            kwargs["worker_init_fn"] = worker_init
        # Session-grouped sequential reading for memmap datasets: sequential
        # multi-GB/s reads instead of random-access thrash, with
        # block-level shuffle preserving training randomness. Only applies
        # when the dataset exposes its session structure.
        if (
            shuffle
            and self.session_shuffle
            and callable(getattr(dataset, "session_index_groups", None))
        ):
            sampler = SessionShuffleSampler(
                dataset.session_index_groups(),
                batch_size=self.batch_size,
                block_batches=self.session_block_batches,
                seed=int(os.environ.get("PL_GLOBAL_SEED", "0") or 0),
            )
            kwargs["sampler"] = sampler
            kwargs["shuffle"] = False
        if loader_num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
            if loader_multiprocessing_context is not None:
                kwargs["multiprocessing_context"] = loader_multiprocessing_context
        return DataLoader(**kwargs)

    @staticmethod
    def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Custom collate function to handle variable-length keystroke labels and dataset_name."""
        import torch
        from torch.utils.data._utils.collate import default_collate

        debug = _debug_steps_enabled()
        t0 = time.perf_counter()
        rank = os.environ.get("RANK", "0")
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else "main"

        if not batch:
            return {}

        # Pop keystroke_labels when ANY sample carries them (avoids silently
        # dropping the field in mixed batches whose first sample lacks it).
        # Samples without the field contribute None so the per-sample list
        # stays aligned; downstream consumers already skip None entries.
        has_keystroke = any("keystroke_labels" in sample for sample in batch)
        keystroke_labels = (
            [sample.pop("keystroke_labels", None) for sample in batch]
            if has_keystroke
            else []
        )
        dataset_names = [sample.pop("dataset_name", "unknown")
                        for sample in batch]

        # Keep only keys present in ALL samples (handles mixed-dataset batches)
        common_keys = set(batch[0].keys())
        for sample in batch[1:]:
            common_keys &= set(sample.keys())
        stripped_batch = [{k: sample[k] for k in common_keys} for sample in batch]

        # Use default collate for other fields
        collated = default_collate(stripped_batch)

        # Add keystroke_labels and dataset_name as lists (not stacked)
        collated["keystroke_labels"] = keystroke_labels
        collated["dataset_name"] = dataset_names

        if debug:
            emg = collated.get("emg")
            print(
                f"[emg2pose-debug][rank={rank}][worker={worker_id}] collate "
                f"n={len(batch)} emg={tuple(emg.shape) if emg is not None else None} "
                f"{time.perf_counter() - t0:.3f}s",
                flush=True,
            )

        return collated

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.train_dataset = self._build_dataset("train", self.train_transforms)
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.val_dataset = self._build_dataset("val", self.val_transforms)
        if len(self.val_dataset) == 0:
            return []
        return self._make_loader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.eval_num_workers,
            multiprocessing_context=self.eval_multiprocessing_context,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            self.test_dataset = self._build_dataset("test", self.test_transforms)
        if len(self.test_dataset) == 0:
            return []
        return self._make_loader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.eval_num_workers,
            multiprocessing_context=self.eval_multiprocessing_context,
        )
