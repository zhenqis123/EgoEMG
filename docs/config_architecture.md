# Hydra Config Architecture (L0 / L1 / L2)

This document describes the three-layer Hydra config structure introduced to
keep experiments maintainable and parameter overrides unambiguous.

## The three layers

```
config/base.yaml                          ← L0: framework skeleton
└── config/lineage/{emgformer,fusion,
                    classic}.yaml         ← L1: per-main-line shared defaults
    └── config/experiment/{emgformer,
        fusion,emg2pose}/*.yaml           ← L2: single experiment, deltas only
```

### L0 — `config/base.yaml` (framework skeleton)

Holds only what every training run needs regardless of model family:

- `defaults` lists the framework Hydra groups (`dataset`, `datamodule`,
  `optimizer`, `lr_scheduler`, `data_split`, `transforms`) plus a placeholder
  `experiment` entry so `experiment=<group>/<name>` CLI overrides resolve.
- Global switches: `seed`, `train`, `eval`, `checkpoint`, `pretrained_*`,
  `freeze_backbone`, `component_lr_scales`, `ignore_head_tail_dims`.
- Monitor / loss framework: `monitor_metric`, `monitor_mode`, base
  `loss_weights` (`{mae: 1, landmark/fingertip: 0.01}`).
- Trainer / callbacks / logger / hydra output-dir skeletons.

It deliberately does **not** pin: `window_length`, `stride`, `lr`, `devices`,
`max_epochs`, `precision`, model/data-specific paths. Those belong to a lineage
or an experiment. The placeholder default experiment
(`emgformer/regression_egoemg`) exists only so Hydra's defaults list has an
`experiment` group to override — always pass `experiment=...` explicitly.

### L1 — `config/lineage/<lineage>.yaml` (shared defaults per main line)

Each main line has one lineage file that pins **only parameters verified
UNIFORM across all experiments in that line**. Dispersed parameters stay on
experiments so inheriting a lineage never changes an experiment's resolved
config.

| Lineage | Pins (uniform across line) | Leaves to experiments |
|---|---|---|
| `emgformer.yaml` | precision=bf16-mixed, gradient_clip_val=1, check_val_every_n_epoch=1, matmul_precision=high, ignore_head_tail_dims=0; defaults group selection (module=emgformer, featurizer=tds_slim, dataset=egoemg_unified_angle_regression, datamodule=egoemg, transforms=emgformer_regression_aug_extended, augmentation=batch_aug, lr_scheduler=cosine) | window_length, stride, lr, max_epochs, decoder preset, log_every_n_steps, devices, callbacks, logger |
| `fusion.yaml` | precision=bf16-mixed, gradient_clip_val=1, matmul_precision=high, ignore_head_tail_dims=0; defaults group selection (dataset=egoemg_angle_regression, datamodule=default, lr_scheduler=cosine) | module (mid_fusion/resnet_vision/vit_vision/wilor_vit), transforms, window_length, stride, dataset_repeat, optimizer, loss_weights, vision fields, max_epochs, devices |
| `emg2pose.yaml` | matmul_precision=high, ignore_head_tail_dims=0, max_epochs=100, check_val_every_n_epoch=1, log_every_n_steps=50; defaults group selection | module (pose/pose_stateful, each inlines its own `network:` block), precision, gradient_clip_val, datamodule.{wl,stride,norm}, lr, devices |

### L2 — `config/experiment/<group>/<name>.yaml` (single experiment)

Expresses only the delta from its lineage. Conventions:

- `defaults` first entry is `- /lineage/<lineage>`.
- Child experiments that inherit another experiment (e.g.
  `regression_egoemg_incre_small` inherits `regression_egoemg`) get the lineage
  through their parent — they should **not** add `/lineage/...` themselves.
- Write only what differs from the lineage: special window_length/stride,
  decoder preset, optimizer, callbacks (filename pattern), logger name,
  vision paths, etc.

## Authoring a new experiment

1. Pick the lineage matching your main line.
2. Create `config/experiment/<group>/<name>.yaml` starting from:

   ```yaml
   # @package _global_
   defaults:
     - /lineage/<lineage>
     - _self_

   # Only your deltas:
   datamodule:
     window_length: 12_000   # if different from lineage default
   optimizer:
     lr: 6.0e-4
   trainer:
     max_epochs: 200
     devices: [0, 1]
   callbacks:
     - _target_: pytorch_lightning.callbacks.ModelCheckpoint
       monitor: ${monitor_metric}
       mode: ${monitor_mode}
       save_top_k: 3
       save_last: true
       filename: "myexp-{epoch:03d}-{${monitor_metric}:.4f}"
   logger:
     _target_: pytorch_lightning.loggers.TensorBoardLogger
     save_dir: logs/<area>
     name: <my_exp_name>
     version: null
     default_hp_metric: false
   ```

3. Run `python scripts/migrate/compare_resolved.py snapshot-one experiment=<group>/<name>`
   to inspect the fully-resolved config and confirm it matches intent.

## Why lineage pins only uniform params

Earlier attempts tried to pin "mainstream" values (e.g. window_length=12000 for
emgformer). But each main line has real dispersion (emgformer has wl=7790 /
12000 / 14638; fusion has stride=400/780/1200/1560 across 4 module families).
Pinning a single "mainstream" value silently changed any experiment that used a
different value. The strict rule **pin only what every experiment already
agrees on** guarantees migration is behavior-preserving, verified by
`scripts/migrate/compare_resolved.py` snapshot diffs (every migrated experiment
resolved identically to its pre-migration state, apart from the harmless
`lineage:` identifier key).

## Migration tooling

- `scripts/migrate/compare_resolved.py` — composes each experiment via the same
  path as `egoemg.train`, flattens the resolved config to `{key: value}`, and
  diffs against a baseline snapshot. Subcommands: `snapshot`, `verify-one`,
  `diff`, `snapshot-one`.
- `scripts/migrate/migrate_experiments.py` — batch-migrates a main line's
  experiments to inherit their lineage (text-preserving edit, then per-file
  `verify-one`).
