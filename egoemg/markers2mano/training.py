"""Training components for the released 21-marker global-head architecture.

The historical trainer source is no longer available. This ports the surviving
source augmentation/data code and restores the mesh + all-marker objective;
it is not a claim of exact reproduction of the released checkpoint's run.
"""

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from .augmentation import MarkerAugmentation
from .geometry import (matrix_to_axis_angle, transform_hand_coordinates_torch,
                       transform_joints_coordinates_torch)
from .inference import MODEL_CONFIG
from .model import EfficientGraphTransformer4x, six_d_to_rot_matrix

MARKER_VERTICES = (191, 88, 253, 708, 729, 144, 87, 295, 319, 220, 365,
                   407, 445, 183, 477, 518, 556, 83, 589, 635, 673)
AUGMENT_DEFAULTS = dict(
    marker_noise_std=0.001, noise_dropout_prob=0.1,
    swap_distance_threshold=0.015, swap_prob=0.3, max_swaps_per_frame=3,
    marker_dropout_prob=0.05, max_dropout=3, marker_drift_magnitude=0.005,
    num_drifted_markers=3, bone_scale_range=0.05, spike_prob=0.1,
    spike_scale_min=2.0, spike_scale_max=5.0, marker_self_weight=0.5,
    marker_scale_range=0.4,
)


class MeshMarkerDataset(Dataset):
    """Extract the 21 surface markers from canonical MANO-right mesh supervision."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        verts = sample['verts'].float()
        if verts.shape != (778, 3) or not torch.isfinite(verts).all():
            raise ValueError('Training requires finite (778,3) MANO-right vertices in metres')
        return dict(verts=verts, markers=verts[list(MARKER_VERTICES)])


def resolve_mano_assets_root(root):
    if not root:
        raise FileNotFoundError('Set mano_assets_root to licensed assets containing models/MANO_RIGHT.pkl')
    path = Path(root).expanduser().resolve()
    if not (path / 'models' / 'MANO_RIGHT.pkl').is_file():
        raise FileNotFoundError(f'Missing {path / "models/MANO_RIGHT.pkl"}')
    return path


class Markers2MANOTraining(MarkerAugmentation, pl.LightningModule):
    """Shared network, differentiable MANO decoding, augmentation and geometric loss."""

    def __init__(self, mano_assets_root, lr=1e-5, dropout=0.1,
                 verts_loss_weight=1.0, markers_loss_weight=5.0,
                 skip_augment_prob=0.5, augmentation=None,
                 marker_neighbors_json=None):
        super().__init__()
        augmentation = {**AUGMENT_DEFAULTS, **(augmentation or {})}
        unknown = set(augmentation) - set(AUGMENT_DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown augmentation settings: {sorted(unknown)}')
        if not 0 <= skip_augment_prob <= 1:
            raise ValueError('skip_augment_prob must be between zero and one')
        if not 0 <= augmentation['max_dropout'] < 21:
            raise ValueError('max_dropout must be between zero and twenty')
        self.save_hyperparameters()
        # Flatten augmentation settings for the migrated methods.
        self.hparams.update(augmentation)
        self.hparams.update(MODEL_CONFIG)
        self.hparams.use_centroid_origin = False
        assets = resolve_mano_assets_root(mano_assets_root)
        # Training-only dependency; importing conversion never loads manotorch.
        from manotorch.manolayer import ManoLayer
        self.mano = ManoLayer(use_pca=False, mano_assets_root=str(assets),
                              flat_hand_mean=False)
        self.mano.requires_grad_(False)
        self.model = EfficientGraphTransformer4x(
            num_markers=21, beta_dim=10, embed_dim=1280, num_layers=12,
            num_heads=8, dropout=dropout, ffn_ratio=4)
        self.register_buffer('markers_indices', torch.tensor(MARKER_VERTICES))
        if marker_neighbors_json:
            data = json.loads(Path(marker_neighbors_json).read_text())
        else:
            adjacency = [set() for _ in range(778)]
            for face in self.mano.th_faces.detach().cpu().tolist():
                for vertex in face:
                    adjacency[vertex].update(v for v in face if v != vertex)
            data = {i: sorted(adjacency[i]) for i in MARKER_VERTICES}
        self.marker_neighbor_indices = {int(k): torch.tensor(v, dtype=torch.long)
                                        for k, v in data.items()}
        self._descendants_cache = {child: self._build_descendants(child)
                                   for _, child in self.SKELETON_EDGES}

    def forward(self, markers):
        return self.model(markers)

    def _shared_step(self, batch, batch_idx, stage):
        augment = stage == 'train' and torch.rand(()).item() >= self.hparams.skip_augment_prob
        markers, vertices = self._prepare_input(batch['verts'], batch['markers'], augment=augment)
        vertices = transform_hand_coordinates_torch(vertices, markers)
        markers, _ = transform_joints_coordinates_torch(markers)
        pose6d, betas = self(markers)
        pose = matrix_to_axis_angle(six_d_to_rot_matrix(pose6d.reshape(-1, 16, 6))).reshape(-1, 48)
        # The local marker frame fixes global rotation. No dataset-mean deltas.
        pose = torch.cat((torch.zeros_like(pose[:, :3]), pose[:, 3:]), dim=1)
        mesh = self.mano(pose, betas).verts
        mesh = transform_hand_coordinates_torch(mesh, mesh[:, self.markers_indices])
        predicted_markers, _ = transform_joints_coordinates_torch(mesh[:, self.markers_indices])
        loss_verts = (mesh - vertices).square().mean()
        loss_markers = (predicted_markers - markers).square().mean()
        loss = self.hparams.verts_loss_weight * loss_verts + self.hparams.markers_loss_weight * loss_markers
        metrics = dict(loss_total=loss, loss_verts=loss_verts, loss_markers=loss_markers)
        with torch.no_grad():
            metrics['pve_mm'] = (mesh - vertices).norm(dim=-1).mean() * 1000
            metrics['marker_error_mm'] = (predicted_markers - markers).norm(dim=-1).mean() * 1000
        for name, value in metrics.items():
            self.log(f'{stage}/{name}', value, on_step=stage == 'train', on_epoch=True,
                     prog_bar=name == 'loss_total', sync_dist=True, batch_size=len(markers))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches)
        return dict(optimizer=optimizer,
                    lr_scheduler=dict(scheduler=scheduler, interval='step'))
