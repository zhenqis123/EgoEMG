"""Training migration contracts without the undistributed corpus or licensed MANO."""

import sys
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from egoemg.markers2mano.dataset import HandDatasetSimple
from egoemg.markers2mano.train import main
from egoemg.markers2mano.training import MeshMarkerDataset


def test_missing_training_data_exits_before_model_allocation(tmp_path, capsys):
    config = tmp_path / 'train.yaml'
    config.write_text('data:\n  root: null\n')
    with pytest.raises(SystemExit) as exc:
        main(['--config', str(config)])
    assert exc.value.code == 2
    assert 'training is currently unsupported' in capsys.readouterr().err


@pytest.mark.parametrize('storage', ['h5', 'mmap'])
def test_mesh_only_corpus_reaches_training_batch(tmp_path, storage):
    vertices = np.random.default_rng(12).normal(size=(2, 778, 3)).astype('float32') * .03
    if storage == 'h5':
        (tmp_path / 'hand').mkdir()
        with h5py.File(tmp_path / 'hand/train.h5', 'w') as file:
            file['vertices'] = vertices
    else:
        (tmp_path / 'merged').mkdir()
        np.save(tmp_path / 'merged/train_vertices.npy', vertices)
    dataset = MeshMarkerDataset(HandDatasetSimple(
        str(tmp_path), ['hand'], require_poses=False, require_betas=False))
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample['markers'].shape == (21, 3)
    torch.testing.assert_close(sample['verts'], torch.from_numpy(vertices[1]))
    assert not {'poses', 'betas', 'ava'} & sample.keys()


def test_mmap_missing_required_labels_is_not_silently_accepted(tmp_path):
    (tmp_path / 'merged').mkdir()
    np.save(tmp_path / 'merged/train_vertices.npy', np.zeros((2, 778, 3), dtype='float32'))
    with pytest.raises(FileNotFoundError, match='poses'):
        HandDatasetSimple(str(tmp_path), ['hand'])


def test_training_backward_and_deterministic_validation(monkeypatch, tmp_path):
    from egoemg.markers2mano import training
    from egoemg.markers2mano.model import EfficientGraphTransformer4x

    # A differentiable mesh decoder substitutes only the licensed MANO asset.
    template = torch.randn(778, 3, generator=torch.Generator().manual_seed(9)) * .03

    class Mano(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.register_buffer('th_faces', torch.tensor([[0, 1, 2]]))

        def forward(self, pose, beta):
            mesh = template.unsqueeze(0).expand(len(pose), -1, -1)
            mesh = mesh * (1 + beta[:, :1, None]) + pose[:, 3:6, None].transpose(1, 2) * template
            return SimpleNamespace(verts=mesh)

    package, module = ModuleType('manotorch'), ModuleType('manotorch.manolayer')
    module.ManoLayer = Mano
    monkeypatch.setitem(sys.modules, 'manotorch', package)
    monkeypatch.setitem(sys.modules, 'manotorch.manolayer', module)
    monkeypatch.setattr(training, 'resolve_mano_assets_root', lambda root: tmp_path)
    # Exercise the same 21-marker graph/global head at a small size on CPU.
    monkeypatch.setattr(training, 'EfficientGraphTransformer4x', lambda **kwargs:
                        EfficientGraphTransformer4x(embed_dim=32, num_layers=1,
                                                    num_heads=8, dropout=0))
    model = training.Markers2MANOTraining(str(tmp_path), skip_augment_prob=0)
    monkeypatch.setattr(model, 'log', lambda *args, **kwargs: None)
    batch = dict(verts=template.unsqueeze(0).repeat(2, 1, 1))
    batch['markers'] = batch['verts'][:, training.MARKER_VERTICES]
    model.train()
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    model.eval()
    torch.testing.assert_close(model.validation_step(batch, 0), model.validation_step(batch, 1), rtol=0, atol=0)
