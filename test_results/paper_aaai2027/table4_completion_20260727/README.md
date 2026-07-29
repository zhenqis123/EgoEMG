# AAAI 2027 Table 4 completion: RN50-S and ViT-B-S

Evaluated on 2026-07-27. Lower joint-angle MAE is better. Values below are in
degrees.

## Recommended matched-center results

These numbers use identical center frames for each vision-only/fusion pair.
The split columns are per-subject mean and population standard deviation; Avg
is the sample-weighted mean across all splits and both hands.

| Method | Params | Gesture | User | Both | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| V-RN50 | 23.5M | 4.503 +/- 1.224 | 6.219 +/- 0.247 | 6.098 +/- 0.607 | 5.274 |
| F-RN50+S | 29.4M | 4.369 +/- 1.183 | 6.200 +/- 0.220 | 6.058 +/- 0.607 | 5.184 |
| V-ViT-B | 86.2M | 4.848 +/- 1.271 | 7.017 +/- 0.323 | 6.861 +/- 0.415 | 5.780 |
| F-ViT-B+S | 90.6M | 4.775 +/- 1.226 | 7.032 +/- 0.337 | 6.838 +/- 0.413 | 5.736 |

RN50-S improves Gesture, User, Both, and Avg. ViT-B-S improves Gesture, Both,
and Avg, but slightly degrades User (7.017 -> 7.032 degrees). At the paper's
current one-decimal precision, several small differences disappear; two-decimal
precision is recommended for matched-pair claims.

## Legacy `test_analysis_fusion` results

The requested official script was also run. Its model-specific validation grid
is not identical to the unified center grid and therefore must not be mixed
with the current Table 4 vision-only rows for a matched comparison.

| Method | Gesture | User | Both | Avg |
| --- | ---: | ---: | ---: | ---: |
| F-RN50+S | 4.641 +/- 1.266 | 6.156 +/- 0.575 | 6.019 +/- 0.442 | 5.295 |
| F-ViT-B+S | 5.185 +/- 1.491 | 7.108 +/- 0.607 | 7.017 +/- 0.408 | 5.994 |

Raw artifacts:

- `rn50_s/results.csv` and `rn50_s/eval.log`: official test-analysis output
- `vitb_s/results.csv` and `vitb_s/eval.log`: official test-analysis output
- `vitb_pair_unified.json`: unified ViT-B pair metrics
- `vitb_pair_predictions/`: aligned predictions used for split statistics
- The aligned RN50 pair predictions are under
  `test_results/fusion/7vision_s_crossattn_stopped_20260727/rn50_pair_predictions/`.

Before editing Table 4, choose one protocol for every row. The defensible
choice is the unified matched-center protocol, which requires recomputing the
displayed vision-only split statistics instead of retaining historical
model-specific-grid standard deviations.

## Fully unfrozen RN50-M diagnostic

The earlier `crossattn_trainvision_100e` run was stopped after validation
degraded. Its best checkpoint is epoch 0. Unlike the retained residual model,
all 32.5M parameters are trainable. Unified matched-center evaluation confirms
that the degradation is real and occurs on every split:

| Method | Params | Gesture | User | Both | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| V-RN50 | 23.5M | 4.503 +/- 1.224 | 6.219 +/- 0.247 | 6.098 +/- 0.607 | 5.274 |
| F-RN50+M (fully unfrozen) | 32.5M | 4.551 +/- 1.196 | 6.302 +/- 0.285 | 6.218 +/- 0.627 | 5.345 |

The fully unfrozen fusion is 0.071 degrees (1.35%) worse overall. Its pooled
degradation is 0.048 degrees on Gesture, 0.092 degrees on User, and 0.127
degrees on Both. It should be retained as an ablation/failure analysis, not as
a main Table 4 baseline.

The legacy model-specific-grid `test_analysis_fusion` output is stored in
`rn50_m_full_unfreeze/results.csv`; it reports Avg 5.446 degrees and is subject
to the same grid-mismatch caveat described above.

## Simple frozen-vision RN50-M residual fusion

This experiment uses the original simple `center_supervised` fusion baseline:
temporally pooled EMG and the RN50 global feature are concatenated, and a small
head predicts a residual correction. It has no cross-attention. The RN50
backbone and vision pose head are both frozen, including BN/dropout behavior.

| Method | Params | Gesture | User | Both | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| V-RN50 | 23.5M | 4.503 +/- 1.224 | 6.219 +/- 0.247 | 6.098 +/- 0.607 | 5.274 |
| F-RN50+M (simple, frozen) | 32.1M | 4.391 +/- 1.165 | 6.195 +/- 0.219 | 6.058 +/- 0.602 | 5.197 |

The simple frozen model improves overall MAE by 0.077 degrees (1.45%). It
improves every split and is only 0.013 degrees behind the more complex RN50-S
cross-attention result (5.197 vs 5.184 degrees). This is the cleaner main-table
candidate because it matches the fusion architecture currently described in
the paper.
