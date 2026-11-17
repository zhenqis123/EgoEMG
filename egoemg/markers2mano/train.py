"""Training entrypoint. The required processed training corpus is not distributed."""

import argparse
from pathlib import Path

import yaml


def validate_inputs(config):
    """Fail before allocating the network when the external training inputs are absent."""
    data = config.get('data', {})
    if not data.get('root'):
        raise FileNotFoundError(
            'Markers2MANO training is currently unsupported: the processed mesh '
            'training corpus is not provided. EgoEMG captures are not a substitute. '
            'See egoemg/markers2mano/README.md#training-source.')
    root = Path(data['root']).expanduser()
    for split in ('train', 'val'):
        mmap = root / 'merged' / f'{split}_vertices.npy'
        if data.get('use_mmap', True) and mmap.is_file():
            continue
        names = data.get('names', [])
        if not names:
            raise ValueError('data.names must list processed corpus subdirectories')
        for name in names:
            path = root / name / f'{split}.h5'
            if not path.is_file():
                raise FileNotFoundError(f'Training corpus is not available: {path}')
    if config.get('resume_checkpoint') and config.get('warm_start_checkpoint'):
        raise ValueError('Choose either resume_checkpoint or warm_start_checkpoint')
    from .training import resolve_mano_assets_root
    resolve_mano_assets_root(config.get('mano_assets_root'))
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True,
                        help='Training YAML; see config/markers2mano/train.yaml (data unavailable)')
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    try:
        root = validate_inputs(config)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader
    from .dataset import HandDatasetSimple
    from .inference import load_model
    from .training import Markers2MANOTraining, MeshMarkerDataset

    pl.seed_everything(config.get('seed', 42), workers=True)
    torch.set_float32_matmul_precision('high')
    data = config['data']
    loaders = []
    loader_config = config.get('loader', {})
    workers = int(loader_config.get('num_workers', 4))
    for split in ('train', 'val'):
        dataset = MeshMarkerDataset(HandDatasetSimple(
            str(root), data.get('names', []), split=split,
            require_poses=False, require_betas=False,
            use_mmap=data.get('use_mmap', True)))
        if len(dataset) == 0:
            raise ValueError(f'The {split} split is empty')
        loaders.append(DataLoader(
            dataset, batch_size=int(loader_config.get('batch_size', 32)),
            num_workers=workers, shuffle=split == 'train',
            pin_memory=True, persistent_workers=workers > 0))
    model = Markers2MANOTraining(config['mano_assets_root'], **config.get('model', {}))
    if config.get('warm_start_checkpoint'):
        network = load_model(config['warm_start_checkpoint'])
        model.model.load_state_dict(network.state_dict(), strict=True)
        del network
    logger = TensorBoardLogger(config.get('output_dir', 'logs/markers2mano'), name='train')
    trainer = pl.Trainer(
        **config.get('trainer', {}), logger=logger,
        callbacks=[LearningRateMonitor(logging_interval='step'), ModelCheckpoint(
            monitor='val/loss_total', mode='min', save_top_k=1, save_last=True,
            filename='markers2mano-{epoch:03d}', auto_insert_metric_name=False)])
    trainer.fit(model, *loaders, ckpt_path=config.get('resume_checkpoint'))


if __name__ == '__main__':
    main()
