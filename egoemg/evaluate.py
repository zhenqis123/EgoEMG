"""Evaluate EgoEMG checkpoints with native temporal supervision."""
from __future__ import annotations

import argparse
import time
import zipfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from egoemg.constants import JOINTS
from egoemg.training.datamodule import WindowedEmgDataModule
from egoemg.datasets.session.emg import (EgoEmgSessionDataset)
from egoemg.datasets.session.schema import (_array)
from egoemg.datasets.session.fullframe import (EgoEmgSessionFullFrameDataset)
from egoemg.datasets.session.fusion import (
    EgoEmgSessionFullFrameFusionDataset,
)
from egoemg.datasets.session.temporal import (
    EgoEmgSessionTemporalVisionDataset,
)
from egoemg.datasets.session.fusion import (
    EgoEmgSessionTemporalFusionDataset,
)
from egoemg.datasets.session_batch import SessionBatchDataset
from egoemg.training.prediction import EmgPredictionModule

ROOT = Path(__file__).resolve().parents[1]
CLASSIC_EMG = {'emg2pose', 'vemg2pose', 'neuropose', 'sensingdynamics'}


def is_emg_model(name):
    return name.startswith('emgformer_') or name in CLASSIC_EMG


SPLITS = {1: 'user', 2: 'gesture', 3: 'both', 4: 'recent_object'}
JOINT_NAMES = [j.name for j in JOINTS] + ['WRIST_FE', 'WRIST_DEVIATION']


def model_config(checkpoint, name, directory):
    """Resolve saved architecture and replace external branch initializers locally."""
    conf = OmegaConf.create(checkpoint['hyper_parameters']['module_conf'])
    # Saved EMG head widths can reference the top-level module decoder.
    context = OmegaConf.create({
        'module': conf, 'temporal_num_frames': 17,
        'temporal_vision_checkpoint': None})
    if name.startswith('fusion_'):
        backbone = name.removeprefix('fusion_').removesuffix('_emgformer_s')
        context.module.vision_checkpoint = str(directory/f'vision_{backbone}.ckpt')
        context.module.emg_checkpoint = str(directory/'emgformer_s.ckpt')
    conf = OmegaConf.create(OmegaConf.to_container(context.module, resolve=True))
    if 'pretrained' in conf:
        conf.pretrained = False
    if conf.get('vision_config') is not None:
        conf.vision_config.pretrained = False
    return conf


class Totals:
    def __init__(self):
        self.error = np.zeros(22, np.float64)
        self.count = np.zeros(22, np.int64)
        self.samples = 0
        self.timepoints = 0

    def add(self, error, count, timepoints, samples=1):
        self.error += error
        self.count += count
        self.samples += samples
        self.timepoints += int(timepoints)

    def metrics(self):
        def ratio(indices):
            n = self.count[indices].sum()
            return self.error[indices].sum()/n if n else np.nan
        return dict(samples=self.samples, valid_timepoints=self.timepoints,
                    valid_angles=int(self.count.sum()), mae=ratio(slice(None)),
                    finger_mae=ratio(slice(0, 20)), wrist_mae=ratio(slice(20, 22)))


class Report:
    """Accumulate errors/counts; all merged rows are valid-angle weighted."""
    def __init__(self):
        self.splits = defaultdict(Totals)
        self.groups = defaultdict(Totals)

    def add(self, split, hand, subject, session, error, count, timepoints):
        for axis in [split, 'all']:
            for side in [hand, 'both']:
                self.splits[axis, side].add(error, count, timepoints)
                for kind, value in [('subject', subject), ('session', session)]:
                    self.groups[kind, str(value), axis, side].add(
                        error, count, timepoints)

    def add_group(self, kind, value, split, hand, error, count, timepoints):
        for axis in [split, 'all']:
            for side in [hand, 'both']:
                self.groups[kind, str(value), axis, side].add(
                    error, count, timepoints)

    def write(self, output):
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(split=split, hand=hand, **s.metrics())
                      for (split, hand), s in sorted(self.splits.items())
                      ]).to_csv(output/'results.csv', index=False)
        rows = []
        for (split, hand), s in sorted(self.splits.items()):
            for j, name in enumerate(JOINT_NAMES):
                rows.append(dict(split=split, hand=hand, joint=name,
                                 valid_angles=int(s.count[j]),
                                 mae=s.error[j]/s.count[j] if s.count[j] else np.nan))
        pd.DataFrame(rows).to_csv(output/'joint_results.csv', index=False)
        pd.DataFrame([dict(group_type=kind, group=value, split=split, hand=hand,
                           **s.metrics())
                      for (kind, value, split, hand), s in sorted(self.groups.items())
                      ]).to_csv(output/'group_results.csv', index=False)


def datasets_for(name, role, checkpoint, args):
    emg = is_emg_model(name)
    common = dict(dataset_dir=args.dataset_dir, allowed_splits=list(SPLITS.values()),
                  evaluation_partition_path=args.partition,
                  evaluation_role=role)
    stats = ROOT/'assets/egoemg_emg_normalization_filtered.json'
    if emg:
        dm = checkpoint['hyper_parameters']['datamodule']
        window = int(dm['val_test_window_length'])
        stride = int(dm['val_test_stride'])
        field = ('emg_raw' if '_raw_' in name or name == 'sensingdynamics'
                 else 'emg_filtered')
        if field == 'emg_raw':
            stats = ROOT/'assets/egoemg_emg_normalization_raw.json'
        return [EgoEmgSessionDataset(
            **common, target_hand=hand, window_length=window, stride=stride,
            emg_field=field, norm_mode='per-dataset', norm_stats_path=stats,
            max_open_sessions=32, jitter=False) for hand in ['left', 'right']]
    temporal = '_temporal_' in name
    fusion = name.startswith('fusion_')
    cls = (EgoEmgSessionTemporalFusionDataset if temporal and fusion else
           EgoEmgSessionTemporalVisionDataset if temporal else
           EgoEmgSessionFullFrameFusionDataset if fusion else
           EgoEmgSessionFullFrameDataset)
    common.update(sample_interval_us=400000, max_open_sessions=96)
    if temporal:
        common.update(num_frames=17, frame_interval_us=250000,
                      context_cache_dir=args.dense_cache,
                      context_partition_path=args.partition)
    if fusion:
        common.update(window_length=8000, norm_stats_path=stats, jitter=False)
    return [cls(**common)]


class Metadata:
    def __init__(self, dataset):
        self.emg = isinstance(dataset, EgoEmgSessionDataset)
        self.dataset = dataset
        self.records = ({r.session_id: r for r in dataset._sessions} if self.emg else
                        {r['session']['session_id']: r for r in dataset.records})
        self.arrays = {}

    def row(self, batch, index, side, length, left_context):
        session = batch['episode_id'][index]
        record = self.records[session]
        hand = ['left', 'right'][side]
        if self.emg:
            folder, arrays = record.folder, record.arrays
            start = int(batch['emg_start_row'][index])+left_context
            selection = slice(start, start+length)
            subject, source = record.subject, record.source_group
        else:
            s = record['session']
            subject, source = str(s['subject']), s['source_group']
            payload = s['hands'].get(hand)
            if payload is None:
                return session, subject, source, {}
            folder, arrays = record['folder'], payload['arrays']
            frame = int(batch['camera_frame'][index])
            local = np.searchsorted(record['alignment']['frames'], frame)
            selection = int(record['alignment']['rows'][local, side])
        labels = {}
        for key in ['object_id', 'gesture_class', 'timestamps']:
            if key not in arrays:
                continue
            cache_key = (session, hand, key)
            if cache_key not in self.arrays:
                self.arrays[cache_key] = _array(folder, arrays[key])
            labels[key] = np.atleast_1d(self.arrays[cache_key][selection])
        return session, subject, source, labels


def accumulate_batch(report, prediction, target, valid, batch, metadata, module):
    """GPU reductions keep full-window EMG outputs off the CPU metrics path."""
    bimanual = module.model.bimanual_output if hasattr(
        module.model, 'bimanual_output') else False
    error = (prediction.float()-target.float()).abs()
    if valid.ndim == 2:
        valid = valid[:, None, :].expand_as(error)
    valid = valid.bool().clone()
    items = []
    for row in range(len(error)):
        index, side = (row//2, row % 2) if bimanual else (
            row, 0 if metadata.dataset.target_hand == 'left' else 1)
        item = metadata.row(batch, index, side, error.shape[-1],
                            int(module.model.left_context))
        if item[2] == 'showee':
            valid[row, 20:] = False
        items.append((index, side, item))
    error = error.masked_fill(~valid, 0.)
    sums = error.double().sum(-1).cpu().numpy()
    counts = valid.sum(-1).cpu().numpy()
    points = valid.any(1).sum(-1).cpu().numpy()
    for row, (index, side, item) in enumerate(items):
        session, subject, _, labels = item
        split_id = int(batch['split_id'][index, side] if bimanual
                       else batch['split_id'][index])
        if split_id not in SPLITS or counts[row].sum() == 0:
            continue
        split, hand = SPLITS[split_id], ['left', 'right'][side]
        report.add(split, hand, subject, session, sums[row], counts[row], points[row])
        for kind, key in [('object', 'object_id'), ('gesture', 'gesture_class')]:
            ids = labels.get(key, np.array([-1]))
            unique = np.unique(ids)
            if len(unique) == 1:
                report.add_group(kind, unique[0], split, hand,
                                 sums[row], counts[row], points[row])
            else:
                for value in unique:
                    select = torch.as_tensor(ids == value, device=error.device)
                    e = error[row, :, select].double().sum(-1).cpu().numpy()
                    c = valid[row, :, select].sum(-1).cpu().numpy()
                    t = int(valid[row, :, select].any(0).sum())
                    if c.sum():
                        report.add_group(kind, value, split, hand, e, c, t)
    return valid


def evaluate_one(path, role, args, dataset_cache):
    name = path.stem
    output = args.output_dir/name/role
    outputs = ['results.csv', 'joint_results.csv', 'group_results.csv']
    if args.save_predictions:
        outputs.append('predictions.npz')
    if args.resume and all((output/x).is_file() for x in outputs):
        print(f'SKIP {name}/{role}: complete', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    conf = model_config(checkpoint, name, args.checkpoint_dir)
    module = EmgPredictionModule(
        module_conf=conf, optimizer_conf={'_target_': 'torch.optim.AdamW', 'lr': 1e-4},
        lr_scheduler_conf=None, loss_weights={'mae': 1.},
        enable_fingertip_loss=False)
    module.model.load_state_dict({k.removeprefix('model.'): v
                                 for k, v in checkpoint['state_dict'].items()
                                 if k.startswith('model.')}, strict=True)
    module.to(args.device).eval()
    key = ('emg' if is_emg_model(name) else
           ('temporal_' if '_temporal_' in name else '')+
           ('fusion' if name.startswith('fusion_') else 'vision'), role,
           '_raw_' in name or name == 'sensingdynamics',
           int(checkpoint['hyper_parameters']['datamodule'][
               'val_test_window_length']) if is_emg_model(name) else None)
    if key not in dataset_cache:
        dataset_cache[key] = datasets_for(name, role, checkpoint, args)
    datasets = dataset_cache[key]
    del checkpoint
    batch_size = args.batch_size or (
        16 if '_temporal_' in name else
        32 if 'vit_l' in name else 64 if (name.startswith('fusion_')
                                      or is_emg_model(name))
        else 128)
    report = Report()
    started = time.monotonic()
    archive = (zipfile.ZipFile(output/'predictions.npz', 'w')
               if args.save_predictions else None)
    chunk = 0
    try:
        with torch.inference_mode():
            for dataset in datasets:
                metadata = Metadata(dataset)
                loader = DataLoader(
                    SessionBatchDataset(dataset, WindowedEmgDataModule._collate_fn),
                    batch_size=batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=str(args.device).startswith('cuda'),
                    collate_fn=WindowedEmgDataModule._collate_fn,
                    **({'prefetch_factor': 2} if args.workers else {}))
                for i, batch in enumerate(loader):
                    inputs = {k: v.to(args.device, non_blocking=True)
                              if isinstance(v, torch.Tensor) else v
                              for k, v in batch.items()}
                    autocast = (torch.autocast('cuda', dtype=torch.bfloat16)
                                if str(args.device).startswith('cuda') and
                                args.precision == 'bf16' else nullcontext())
                    with autocast:
                        prediction, target, valid = module(inputs)
                    valid = accumulate_batch(report, prediction, target, valid,
                                             batch, metadata, module)
                    if archive:
                        shape = ((len(batch['episode_id']), 2, 22, -1)
                                 if getattr(module.model, 'bimanual_output', False)
                                 else prediction.shape)
                        values = dict(
                            prediction=prediction.reshape(shape).float().cpu().numpy(),
                            target=target.reshape(shape).float().cpu().numpy(),
                            valid=valid.reshape(shape).cpu().numpy(),
                            session=np.array(batch['episode_id']))
                        for k in ['camera_frame', 'emg_start_row', 'target_hand_index',
                                  'split_id', 'object_id']:
                            if k in batch:
                                values[k] = np.asarray(batch[k])
                        for k, v in values.items():
                            with archive.open(f'{chunk:06d}_{k}.npy', 'w') as f:
                                np.save(f, v, allow_pickle=False)
                        chunk += 1
                    if i % 50 == 0:
                        label = getattr(dataset, 'target_hand', 'frames')
                        print(f'{name}/{role} {label} {i+1}/{len(loader)}',
                              flush=True)
                    if args.max_batches and i+1 >= args.max_batches:
                        break
    finally:
        if archive:
            archive.close()
    if not report.splits:
        raise RuntimeError(f'No valid evaluation entries: {name}/{role}')
    report.write(output)
    mae = report.splits['all', 'both'].metrics()['mae']
    print(f'DONE {name}/{role}: MAE={mae:.8f}; '
          f'{time.monotonic()-started:.1f}s', flush=True)
    del module
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path,
                        default=ROOT/'pretrained_models'/'egoemg')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'logs/evaluation/test_results_new')
    parser.add_argument('--split', nargs='+', choices=['val', 'test'], default=['test'])
    parser.add_argument('--models', nargs='+', help='Checkpoint stems; default all')
    parser.add_argument('--device', default=(
        'cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-batches', type=int,
                        help='Smoke only; use a separate output-dir')
    parser.add_argument('--dataset-dir', type=Path, default=ROOT/'data/EgoEMG_dataset')
    parser.add_argument('--partition', type=Path,
                        help='Default: <dataset-dir>/assets/egoemg_evaluation_protocol.json')
    parser.add_argument('--dense-cache', type=Path,
                        help='Default: <dataset-dir>/fullframe_256_30fps')
    args = parser.parse_args(argv)
    if args.partition is None:
        args.partition = args.dataset_dir/'assets/egoemg_evaluation_protocol.json'
    if args.dense_cache is None:
        args.dense_cache = args.dataset_dir/'fullframe_256_30fps'
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    files = sorted(args.checkpoint_dir.glob('*.ckpt'))
    if args.models:
        files = [p for p in files if p.stem in args.models]
        missing = set(args.models)-{p.stem for p in files}
        if missing:
            parser.error(f'Unknown models: {sorted(missing)}')
    if not files:
        parser.error('No checkpoints selected')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for number, path in enumerate(files, 1):
        for role in dict.fromkeys(args.split):
            print(f'[{number}/{len(files)}] {path.stem}/{role}', flush=True)
            evaluate_one(path, role, args, cache)
    print(f'Completed {len(files)} checkpoints: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
