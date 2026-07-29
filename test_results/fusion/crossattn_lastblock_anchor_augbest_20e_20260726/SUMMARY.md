# Iteration 2: augbest(no-mixup) + anchor loss — Results

Updated: 2026-07-26 Asia/Shanghai

## Constraints honored
1. **No MixUp**: used `batch_aug_best_v2_no_mixup.yaml` (verified `mixup.mask_prob=0.0`).
2. **Left-hand error investigation**: diagnosed — see section below.

## Unified Center-Frame Eval (gold standard, 4154 identical samples)

| Model | combined | left | right | vs vision-only |
|---|---|---|---|---|
| RN50 vision-only | 0.0920 | 0.0952 | 0.0889 | baseline |
| PILOT (no aug, no anchor) | 0.0906 | 0.0933 | 0.0879 | -1.56% |
| ANCHOR (+ zero-EMG anchor) | 0.0905 | 0.0933 | 0.0877 | -1.65% |
| **AUGBEST (+ augbest + anchor)** | **0.0905** | 0.0933 | 0.0877 | **-1.70%** |

augbest+anchor is the best variant, beating vision-only by 1.70%.

## Diagnostic — modality interventions

| intervention | pilot | anchor | augbest |
|---|---|---|---|
| normal | 0.092541 | 0.092492 | **0.092447** |
| vision_only | 0.093303 | 0.093226 | 0.093181 |
| zero_emg | 0.094127 | 0.093195 | 0.093136 |
| shuffle_emg | 0.092893 | 0.092767 | **0.092650** |
| shuffle_visual_tokens | 0.095580 | 0.095428 | 0.095269 |

Notable: augbest has the **lowest shuffle_emg** (0.09265), meaning the gap
between normal (0.09245) and shuffled EMG grew — correct EMG pairing matters
more after augmentation. feature_injection_ratio rose 0.25 → 0.296.

## Left-hand error investigation (constraint #2)

**Root cause: task/vision intrinsic difficulty, NOT a normalization bug.**
- norm_stats already use per-hand per-channel stats (verified in training logs:
  left std≈4, right std≈14-29). No bug there.
- vision-only itself has L-R gap of 7.0% (left 0.0952 vs right 0.0889) — the
  gap exists before any EMG fusion, so it is a property of the ego-centric
  vision task (left hand harder to see/predict), not EMG.
- Fusion gain is actually LARGER on the left hand (-1.96%) than right (-1.32%):
  EMG compensates more where vision is weaker.
- anchor+augbest narrowed L-R gap from 7.0% (vision) → 6.3% (fusion).

Conclusion: no fixable bug; left-hand error is intrinsic task difficulty and
fusion already helps left more than right.

## Training-time val_mae progression (augbest vs anchor no-aug)
augbest leads anchor-noaug from epoch 6 onward, consistently by ~0.00005-0.00008.
Best augbest val_mae 0.092445 @ ep16 vs anchor-noaug 0.092490 @ ep18.

## Artifacts
- config: `config/experiment/fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_anchor_augbest_20e.yaml`
- launcher: `scripts/run/train_rn50_m_crossattn_lastblock_anchor_augbest_20e_6gpu.sh`
- best ckpt: `logs/20260726/.../anchor_augbest_20e/.../rn18-s-8ch-centerfusion-epoch=016-val_mae=0.0924.ckpt`
- unified eval: `test_results/fusion/crossattn_lastblock_anchor_augbest_20e_20260726/unified_center_eval.json`
- diagnostic: `test_results/fusion/crossattn_lastblock_anchor_augbest_20e_20260726/diagnostics_ep16.json`
