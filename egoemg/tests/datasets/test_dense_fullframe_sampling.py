from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from egoemg.training.datamodule import EpochRandomSubsetSampler
from egoemg.datasets.session.alignment import (build_alignment)
from egoemg.datasets.session.fusion import (
    EgoEmgSessionFullFrameFusionDataset,
)
from egoemg.tests.models.test_bimanual_fusion import paired_root  # noqa: F401


def test_sixty_fps_source_produces_complete_thirty_fps_index(tmp_path):
    def store(name, values):
        values = np.asarray(values)
        values.tofile(tmp_path / name)
        return dict(path=name, dtype=str(values.dtype), shape=list(values.shape))

    n = 120
    times = store("times", np.arange(n, dtype=np.int64) * 1_000_000 // 60)
    arrays = dict(timestamps=times,
        split_id=store("split", np.zeros(n, np.int8)),
        joint_angles_valid=store("valid", np.ones(n, np.uint8)),
        joint_angles=store("angles", np.zeros((n, 20), np.float32)))
    physical = np.arange(n, dtype=np.int64) * 3 + 5
    session = dict(hands={"left": dict(arrays=arrays)},
        cameras=dict(showee_head=dict(streams=dict(camera=dict(
            arrays=dict(timestamps=times), video=dict(path="video",
                frame_index_map=store("physical", physical)))))))
    dense = build_alignment(tmp_path, session, interval_us=0, target_fps=30)
    native = build_alignment(tmp_path, session, interval_us=0)
    assert len(native["frames"]) == 120 and len(dense["frames"]) == 60
    np.testing.assert_array_equal(dense["physical"], physical[dense["frames"]])
    np.testing.assert_array_equal(dense["rows"][:, 0], dense["frames"])


def test_global_random_budget_covers_all_pool_and_changes_epochs():
    pool = list(range(101))
    samplers = [EpochRandomSubsetSampler(pool, 50, num_replicas=5, rank=r, seed=42)
                for r in range(5)]
    first = [list(s) for s in samplers]
    assert all(len(x) == 10 for x in first)
    flattened = [x for rank in first for x in rank]
    assert len(set(flattened)) == 50
    assert max(flattened) >= 50  # Not just shuffling the former sparse pool.
    for s in samplers:
        s.set_epoch(1)
    second = [list(s) for s in samplers]
    assert second != first
    assert len(set(x for rank in second for x in rank)) == 50
    single = EpochRandomSubsetSampler(pool, 40, seed=42)
    assert len(list(single)) == len(single) == 40


def test_centered_windows_are_fixed_for_every_selected_image(paired_root):  # noqa: F811
    root, stats = paired_root
    ds = EgoEmgSessionFullFrameFusionDataset(
        root, ["train"], window_length=8, norm_stats_path=stats, jitter=False)
    for i in range(len(ds)):
        first, second = ds[i], ds[i]
        np.testing.assert_array_equal(first["window_offset_samples"], [0, 0])
        np.testing.assert_array_equal(first["emg_start_row"], second["emg_start_row"])
        rid, local = ds.samples[i]
        rows = ds.records[rid]["alignment"]["rows"][local]
        good = first["emg_valid_mask"]
        np.testing.assert_array_equal(first["emg_start_row"][good] + 4, rows[good])


def test_dense30_configuration_uses_random_image_pool_and_no_window_shift():
    with initialize_config_dir(version_base="1.3", config_dir=str(
        Path(__file__).resolve().parents[3] / "config")):
        c = compose(config_name="base", overrides=[
            "experiment=fusion/fusion_vit_s"])
    d = c.dataset.train[0]
    assert d.sample_interval_us == 0 and d.target_fps == 30 and not d.jitter
    assert not c.train_window_jitter
    assert c.datamodule.train_samples_per_epoch == 168928
    assert c.dataset.val[0].sample_interval_us == 400000
    assert c.trainer.max_epochs == c.lr_scheduler.scheduler.T_max == 50
