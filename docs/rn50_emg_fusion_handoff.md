# RN50 + EMG Fusion: Agent Handoff

Updated: 2026-07-26 02:50 Asia/Shanghai

## Objective

Improve EgoEMG center-frame hand-pose accuracy by fusing RN50 vision features
with EMGFormer-M, while preserving the strong RN50 vision-only baseline.

The immediate question is whether controlled visual fine-tuning can improve the
new cross-attention fusion model. Do not redesign the network until the current
20-epoch pilot has been assessed.

## Repository and conventions

- Repository: `. (repo root)`
- Logs are under the repository `logs/` path (a symlink to NVMe storage).
- Hydra hierarchy: `config/base.yaml` → `config/lineage/fusion.yaml` →
  `config/experiment/fusion/*.yaml`.
- Training entrypoint: `python -m emg2pose.train`.
- EgoEMG vision must use all-intra videos and `decord`.
- Primary comparison must eventually use
  `scripts/eval/unified_center_eval.py`, which evaluates identical center frames.
- Training `val_mae` is useful for trends but is not a substitute for the
  unified center-frame result.

## Relevant model

Implementation:

`emg2pose/models/modules/mid_fusion.py`

Fusion mode:

`frozen_cross_attention`

Data flow:

1. RN50 produces frozen or selectively trainable layer3/layer4 feature maps.
2. Trainable 1x1 adapters align both maps to 256 dimensions.
3. EMG TDS tokens query the visual tokens through cross-attention.
4. A pretrained EMGFormer-M decoder processes the conditioned EMG sequence.
5. The center temporal token predicts a residual.
6. Final pose is `vision_pose + residual`.

The RN50 vision checkpoint is:

`logs/fusion/vision_resnet50/version_0/checkpoints/resnet50-vision-epoch=145-val_mae=0.0923.ckpt`

The EMGFormer-M initializer is:

`test_results/egoemg_emgformer_middle_incre_cotrain/checkpoints/best.ckpt`

## Established results (baselines before the cross-attention iterations)

Unified center-frame results from earlier experiments (the baseline numbers the
cross-attention iterations below are measured against):

- RN50 vision-only: `0.092041` (source: `test_results/fusion/unified_center_rn_vision_20260724.json`)
- Best conventional RN50-M fusion: `0.091949` (only about 0.10% better)
- Best RN18-S fusion: `0.094224`
- RN18 vision-only: `0.101942` (RN18 fusion improves about 7.57%)

Note: the unified-center-frame vision-only number (`0.092041`) is the
authoritative baseline; it differs slightly from the training-config `val_mae`
printed on the vision-only checkpoint filename (`0.0923`) because the two use
different center-frame grids. Always compare via `unified_center_eval.py`.

The frozen cross-attention model was diagnosed on its training-config validation
set:

- Normal fusion: `0.092572`
- Vision-only branch: `0.093719`
- Zero EMG: `0.094386`
- Shuffled EMG: `0.093207`
- Shuffled visual tokens: `0.095774`
- Cross-attention disabled: `0.110477`

Interpretation:

- EMG is useful: normal fusion is about 1.22% better than its vision-only branch.
- Correct EMG pairing accounts for about 55.4% of that gain.
- About 44.6% remains after shuffling EMG, so the residual branch also learns a
  visual/distribution shortcut.
- Gradients are not absent from EMGFormer. Optimization is concentrated more
  strongly in the residual head, cross-attention, and visual adapters.

Full diagnostic output:

`test_results/fusion/frozen_crossattn_diagnostics_20260725.json`

Diagnostic script:

`scripts/analysis/diagnose_frozen_cross_attention.py`

## Failed visual-unfreezing experiment

Experiment:

`fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_trainvision_100e`

It unfroze the entire RN50 and vision head at `1e-6`, while other components
used `1e-5`. It was stopped at epoch 5 because validation steadily worsened:

`0.09508 → 0.09705 → 0.09877 → 0.09916 → 0.09981 → 0.10045`

Cause:

- BatchNorm running statistics update independently of the optimizer LR.
- By epoch 5, BN means/variances had drifted about 5%.
- Vision-head dropout was also active in training mode.
- The pretrained visual baseline was therefore damaged despite the small LR.

Do not resume this run.

## Iteration 1: controlled last-block pilot (COMPLETED)

Purpose: test minimal, controlled visual adaptation without changing the fusion
architecture.

Experiment config:

`config/experiment/fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_20e.yaml`

Launcher:

`scripts/run/train_rn50_m_crossattn_lastblock_20e_6gpu.sh`

Log directory:

`logs/20260726/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_20e/train/train/version_0`

Configuration:

- Six GPUs
- 20 epochs
- Per-GPU batch size 300
- EMG/fusion/head LR: `1e-5`
- RN50 final Bottleneck LR: `1e-7`
- Cosine `eta_min`: `5e-8`
- Only `vision_backbone.7.2` is trainable in RN50
- All vision BatchNorm modules are locked in eval mode
- Vision pose head is frozen and locked in eval mode
- Trainable parameters: about 12.4M

Result: best training `val_mae` `0.092536` at epoch 16 (monotone decrease from
`0.092901` at epoch 0). The epoch-0 result confirmed that, unlike full
unfreezing, this scheme preserves the initial visual baseline.

### Unified center-frame eval (gold standard)

The pilot's training `val_mae` (0.0925) is *higher* than the RN50 vision-only
training val (0.0923), but this comparison is unfair: the two models are
evaluated on different center-frame grids. The unified center-frame eval uses a
fixed WL=7790 grid and evaluates every model on the identical 4154 frames.

| Model | unified MAE | vs vision-only |
|---|---|---|
| RN50 vision-only | `0.092041` | baseline |
| Best conventional RN50-M fusion | `0.091949` | `-0.10%` |
| **PILOT lastblock_20e (ep16)** | **`0.090605`** | **`-1.56%`** |

The pilot clearly beats vision-only on the gold-standard eval, and also beats
the best conventional RN50-M fusion by `1.41%`. Artifacts:

- unified eval: `test_results/fusion/crossattn_lastblock_20e_20260726/unified_center_eval.json`
- diagnostic: `test_results/fusion/crossattn_lastblock_20e_20260726/diagnostics_ep16.json`

### Modality intervention diagnostic (pilot ep16)

| intervention | MAE |
|---|---|
| normal | `0.092541` |
| vision_only | `0.093303` |
| zero_emg | `0.094127` |
| shuffle_emg | `0.092893` |
| shuffle_visual_tokens | `0.095580` |
| no_cross_attention | `0.130273` |

Fusion gain over vision-only is `0.82%`, of which `46.2%` comes from correct
EMG pairing. `zero_emg` (`0.0941`) is still *worse* than `vision_only`
(`0.0933`): the residual head still emits a vision-only shortcut when EMG is
absent. `feature_injection_ratio` dropped to `0.251` (frozen model was `0.374`).
This motivated the anchor-loss variant below.

## Iteration 2: zero-EMG anchor loss (COMPLETED)

Motivation: the pilot diagnostic showed the residual head learns a vision-only
shortcut (`zero_emg` worse than `vision_only`). A zero-EMG anchor regularizer
forces the center-frame residual toward zero when the EMG input is zeroed, so
`delta` must encode EMG-driven correction rather than a visual bias.

### Code changes

- `emg2pose/models/modules/mid_fusion.py`: added
  `MidFusionPoseFormer.compute_zero_emg_delta()` — runs the residual branch
  (cross-attn → decoder → head) with zeroed EMG and returns the center-frame
  delta only (no vision_pose addition).
- `emg2pose/lightning.py`: added `anchor_loss_weight` hparam; in `_step`, when
  training and `anchor_loss_weight > 0`, computes
  `(zero_emg_delta ** 2).mean()` and adds it to the loss. Logs
  `train_anchor_l2`.
- `emg2pose/train.py`: passes `anchor_loss_weight` from config to the Lightning
  module.

### Experiment config

`config/experiment/fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_anchor_20e.yaml`
(inherits the lastblock_20e pilot, sets `anchor_loss_weight: 1.0`).

Launcher: `scripts/run/train_rn50_m_crossattn_lastblock_anchor_20e_6gpu.sh`

Note: batch size reduced to 200 (from 300) because the anchor loss adds a
second forward pass per step. This is the cause of the slower throughput
(~2.0 it/s vs 2.5 it/s), not a correctness issue.

Result: best training `val_mae` `0.092490` at epoch 18. Regularization
behaved as designed: `train_anchor_l2` collapsed to ~`5e-6` (zero-EMG delta
suppressed) while `train_delta_l2` stayed at ~`6.8e-3` (real EMG-driven delta
preserved).

### Unified eval + diagnostic (anchor ep18)

| Model | unified MAE | vs vision-only |
|---|---|---|
| **ANCHOR lastblock_anchor_20e (ep18)** | **`0.090486`** | **`-1.65%`** |

Diagnostic key indicator `zero_emg - vision_only`:
- frozen: `+0.00067`
- pilot: `+0.00082`
- **anchor: `-0.00003`** (shortcut essentially eliminated — fusion collapses to
  vision-only when EMG is absent, exactly the intended behavior).

Artifacts:
- unified eval: `test_results/fusion/crossattn_lastblock_anchor_20e_20260726/unified_center_eval.json`
- diagnostic: `test_results/fusion/crossattn_lastblock_anchor_20e_20260726/diagnostics_ep18.json`
- summary: `test_results/fusion/crossattn_lastblock_anchor_20e_20260726/SUMMARY.md`

## Iteration 2b: augbest (no-mixup) + anchor loss (COMPLETED, BEST VARIANT)

Motivation: RN18 fusion experiments benefit substantially from EMG
augmentation. Stack the verified `batch_aug_best_v2_no_mixup` augmentation on
top of the anchor variant. MixUp is disabled (`mask_prob: 0.0`) because mixing
EMG across samples would corrupt the joint-angle targets.

### Experiment config

`config/experiment/fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_anchor_augbest_20e.yaml`
(inherits anchor_20e, adds `override /augmentation: batch_aug_best_v2_no_mixup`).

Launcher: `scripts/run/train_rn50_m_crossattn_lastblock_anchor_augbest_20e_6gpu.sh`

Augmentation groups enabled (mixup disabled): random_gain, mag_warping,
baseline_drift, powerline_noise, channel_mask, time_mask, freq_mask,
gaussian_noise, channel_rotation.

Result: best training `val_mae` `0.092445` at epoch 16. From epoch 6 onward,
augbest consistently led anchor-noaug by ~`0.00005–0.00008` on training val
(aug regularization pays off once the model begins to overfit).

### Unified eval + diagnostic (augbest+anchor ep16) — BEST VARIANT

| Model | unified MAE | vs vision-only |
|---|---|---|
| **AUGBEST lastblock_anchor_augbest_20e (ep16)** | **`0.090475`** | **`-1.70%`** |

This is the best variant overall. Diagnostic shows the lowest `shuffle_emg`
(`0.092650` vs anchor `0.092767`, pilot `0.092893`): augmentation makes the
model rely more on *correct* EMG pairing rather than distribution shortcuts.
`feature_injection_ratio` rose to `0.296`.

Artifacts:
- unified eval: `test_results/fusion/crossattn_lastblock_anchor_augbest_20e_20260726/unified_center_eval.json`
- diagnostic: `test_results/fusion/crossattn_lastblock_anchor_augbest_20e_20260726/diagnostics_ep16.json`
- summary: `test_results/fusion/crossattn_lastblock_anchor_augbest_20e_20260726/SUMMARY.md`

## Left-hand error investigation

The augbest+anchor model has left MAE `0.0933` vs right `0.0877` (left 6.3%
worse). Investigation conclusion: **not a bug, but intrinsic task difficulty.**

- norm_stats already use per-hand per-channel stats (verified in training logs:
  left std≈4, right std≈14-29). No normalization bug.
- vision-only *itself* has a left-right gap of `7.0%` (left `0.0952` vs right
  `0.0889`); the gap exists before any EMG fusion, so it is a property of the
  ego-centric vision task (left hand harder to see/predict), not EMG.
- Fusion gain is actually *larger* on the left hand (`-1.96%`) than right
  (`-1.32%`): EMG compensates more where vision is weaker.
- anchor+augbest narrowed the L-R gap from `7.0%` (vision) to `6.3%` (fusion).

No fix to apply; left-hand error is intrinsic and fusion already helps left
more than right.

## Current state of results

Unified center-frame eval (gold standard, 4154 identical samples):

| Model | unified MAE | left | right | vs vision-only |
|---|---|---|---|---|
| RN50 vision-only | `0.092041` | `0.095153` | `0.088907` | baseline |
| Best conventional RN50-M fusion | `0.091949` | — | — | `-0.10%` |
| PILOT lastblock_20e (ep16) | `0.090605` | `0.093312` | `0.087879` | `-1.56%` |
| ANCHOR lastblock_anchor_20e (ep18) | `0.090486` | `0.093279` | `0.087692` | `-1.65%` |
| **AUGBEST lastblock_anchor_augbest_20e (ep16)** | **`0.090475`** | `0.093287` | `0.087663` | **`-1.70%`** |

All three cross-attention variants clearly beat vision-only. augbest+anchor is
the best. The cross-attention fusion line is now the strongest fusion result on
this dataset, surpassing the conventional RN50-M fusion by `1.41%`.

## Decision criteria (status)

Original success criteria, all met:

- [x] Unified MAE below RN50 vision-only (`0.092041`). Best variant: `0.090475`.
- [x] Lower than conventional RN50-M fusion (`0.091949`). Best variant beats it
      by `1.41%`.
- [x] Correctly paired EMG outperforms shuffled/zero EMG (normal `0.092447` <
      shuffle_emg `0.092650` < vision_only `0.093181`).
- [x] BN running statistics and frozen vision-head weights unchanged (verified
      by epoch-0 val preserving the baseline).

## Suggested next directions

Ranked by expected value, with evidence:

1. **Boost EMG utilization.** `feature_injection_ratio` is only `0.296` and
   `shuffle_emg` barely changes the result — EMG is under-exploited. Options:
   raise the cross-attention LR (`early_fusion.cross_attention` currently shares
   the `1e-5` base LR via `component_lr_scales`); deepen the fusion
   (`cross_attention_ffn_dim` 512→1024 or a second cross-attn layer); unfreeze
   the last 1–2 decoder layers (decoder `g/w` was the lowest at `0.0009`).
2. **Longer schedule.** augbest ep16 was still slowly improving; a 50-epoch run
   with `eta_min=5e-6` (like the 100e config) may yield another `0.3–0.5%`.
3. **Learned multi-scale weighting.** layer3+layer4 are summed equally in
   `FrozenVisualCrossAttention`; a learnable weight could help (evidence:
   `shuffle_visual_tokens` still costs `0.0953`).
4. **Token self-attention mode.** `JointTokenFusionEncoder` (already
   implemented, unused on RN50-M) is a stronger fusion architecture; worth a
   direct comparison.

Do NOT repeat the full-unfreezing experiment (`..._trainvision_100e`); it
destroys the BatchNorm baseline.
