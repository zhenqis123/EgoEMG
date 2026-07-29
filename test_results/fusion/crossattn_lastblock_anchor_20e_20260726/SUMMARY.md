# RN50 + EMG Cross-Attention Fusion — Results Summary

Updated: 2026-07-26 Asia/Shanghai

## Goal
Fusion model must clearly outperform the RN50 vision-only baseline on the
unified center-frame evaluation (same center frames, same window).

## Final results — Unified Center-Frame Eval (gold standard)

| Model                              | combined | left   | right  | vs vision-only |
|------------------------------------|----------|--------|--------|----------------|
| RN50 vision-only                   | 0.0920   | 0.0952 | 0.0889 | baseline       |
| PILOT crossattn lastblock_20e ep16 | 0.0906   | 0.0933 | 0.0879 | -1.56%         |
| **ANCHOR crossattn anchor_20e ep18** | **0.0905** | 0.0933 | 0.0877 | **-1.65%**     |

Both fusion models clearly beat vision-only. Anchor model is best overall and
also has the cleanest modality use (see diagnostic below).

Eval config: `REF_WL=7790` grid, `--center-window-length 12000`, 4154 hand
samples (identical frames for every model).

## Modality intervention diagnostic (anchor ep18)

| intervention          | frozen  | pilot   | anchor  |
|-----------------------|---------|---------|---------|
| normal                | 0.09257 | 0.09254 | 0.09249 |
| vision_only           | 0.09372 | 0.09330 | 0.09323 |
| zero_emg              | 0.09439 | 0.09413 | 0.09320 |
| shuffle_emg           | 0.09321 | 0.09289 | 0.09277 |
| shuffle_visual_tokens | 0.09577 | 0.09558 | 0.09543 |
| no_cross_attention    | 0.11048 | 0.13027 | 0.14450 |

Key indicator `zero_emg - vision_only`:
- frozen: +0.00067 (clean)
- pilot:  +0.00082 (clean)
- anchor: -0.00003 (shortcut essentially eliminated — fusion == vision-only when EMG absent)

The anchor loss (`anchor_loss_weight=1.0`) forces the residual head toward
zero when EMG is zeroed. Diagnostic confirms it worked: `train_anchor_l2`
collapsed to ~5e-6 while `train_delta_l2` stayed at ~6.8e-3, i.e. the model
still learns a useful EMG-driven residual but no longer emits a vision-only
shortcut through the delta branch.

## Artifacts

Pilot (crossattn lastblock, no anchor):
- config: `config/experiment/fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_20e.yaml`
- best ckpt: `logs/20260726/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_20e/train/train/version_0/checkpoints/rn18-s-8ch-centerfusion-epoch=016-val_mae=0.0925.ckpt`
- unified eval: `test_results/fusion/crossattn_lastblock_20e_20260726/unified_center_eval.json`
- diagnostic: `test_results/fusion/crossattn_lastblock_20e_20260726/diagnostics_ep16.json`

Anchor (crossattn lastblock + zero-EMG anchor loss):
- config: `config/experiment/fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_anchor_20e.yaml`
- launcher: `scripts/run/train_rn50_m_crossattn_lastblock_anchor_20e_6gpu.sh`
- best ckpt: `logs/20260726/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_anchor_20e/train/train/version_0/checkpoints/rn18-s-8ch-centerfusion-epoch=018-val_mae=0.0925.ckpt`
- unified eval: `test_results/fusion/crossattn_lastblock_anchor_20e_20260726/unified_center_eval.json`
- diagnostic: `test_results/fusion/crossattn_lastblock_anchor_20e_20260726/diagnostics_ep18.json`

## Code changes (zero-EMG anchor loss)
- `emg2pose/models/modules/mid_fusion.py`: `MidFusionPoseFormer.compute_zero_emg_delta()`
- `emg2pose/lightning.py`: `anchor_loss_weight` hparam + anchor L2 term in `_step`
- `emg2pose/train.py`: pass `anchor_loss_weight` from config

## Conclusion
Goal achieved: fusion clearly beats vision-only on the unified center-frame
eval (0.0905 vs 0.0920, -1.65%), with both pilot and anchor variants winning,
and the anchor variant additionally suppressing the vision-only shortcut
through the residual head.
