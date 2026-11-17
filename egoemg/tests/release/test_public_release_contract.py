"""Public recipe/CLI contract shared by main and release; no training runs."""
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[3]
RECIPES = (
    'emgformer/emgformer_s',
    'emgformer/emg2pose_emgformer_s_stride390',
    'emgformer/emg2pose_emgformer_m_stride390',
    'emgformer/emg2pose_emgformer_l_stride390',
    'fusion/vision_vit_s',
    'fusion/fusion_vit_s',
    'fusion/direct_fusion_vit_s',
    'fusion/temporal_vision_vit_s',
    'fusion/temporal_fusion_vit_s',
)


@pytest.mark.parametrize('recipe', RECIPES)
def test_public_recipe_selects_validation_only(recipe):
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'config')):
        cfg = compose(config_name='base', overrides=['experiment=' + recipe])
    assert cfg.train and not cfg.eval
    assert all(list(x.allowed_splits) == ['train'] for x in cfg.dataset.train)
    for role in ('val', 'test'):
        for dataset in cfg.dataset[role]:
            if dataset._target_ == 'egoemg.datasets.corpora.emg2pose.Emg2PoseDataset':
                assert list(dataset.allowed_splits) == [role]
                assert dataset.skip_ik_failures
            else:
                assert dataset.evaluation_role == role
                assert dataset.evaluation_partition_path.endswith(
                    'egoemg_evaluation_protocol.json')
    assert all(x.monitor.startswith('val_') for x in cfg.callbacks if x.get('monitor'))


@pytest.mark.parametrize('module', (
    'egoemg.train', 'egoemg.evaluate', 'egoemg.markers2mano',
    'egoemg.markers2mano.train',
    'scripts.data.session.build_sparse_fullframe_cache',
    'scripts.data.session.build_dense_fullframe_cache',
))
def test_public_cli_help_without_data(module):
    result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]


def test_public_paper_recipes_are_available():
    experiment_root = ROOT / 'config/experiment'
    actual = {path.relative_to(experiment_root).with_suffix('').as_posix()
              for path in experiment_root.rglob('*.yaml')}
    assert set(RECIPES) <= actual


@pytest.mark.parametrize('size,layers,model_dim', [
    ('s', 3, 256), ('m', 6, 256), ('l', 8, 384),
])
def test_public_emg2pose_preserves_stride390_update_and_selection_protocol(
        size, layers, model_dim):
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'config')):
        cfg = compose(config_name='base', overrides=[
            f'experiment=emgformer/emg2pose_emgformer_{size}_stride390'])
    assert cfg.datamodule.stride == cfg.dataset.train[0].stride == 390
    assert cfg.datamodule.window_length == cfg.datamodule.val_test_stride == 7790
    assert cfg.module.out_channels == cfg.module.head.out_channels == 20
    assert not cfg.module.decoder.causal
    assert cfg.module.decoder.num_layers == layers
    assert cfg.module.decoder.model_dim == model_dim
    assert cfg.enable_fingertip_loss
    assert cfg.loss_weights['landmark/fingertip'] == 0.01
    assert cfg.validation_user_macro and cfg.monitor_metric == 'val_user_macro_mae'
    assert all(callback.get('monitor', cfg.monitor_metric) == cfg.monitor_metric
               for callback in cfg.callbacks)
    effective_batch = (cfg.batch_size * len(cfg.trainer.devices)
                       * cfg.trainer.accumulate_grad_batches)
    assert effective_batch == 3000
    assert cfg.trainer.max_steps == 33600
    assert (cfg.trainer.val_check_interval
            // cfg.trainer.accumulate_grad_batches) == 224
    assert cfg.lr_scheduler.interval == 'step'
    assert cfg.lr_scheduler.scheduler.steps_per_period == 224
    assert cfg.lr_scheduler.scheduler.T_max == 150
    assert cfg.optimizer.lr == 5e-4
    assert cfg.lr_scheduler.scheduler.eta_min == 5e-5
