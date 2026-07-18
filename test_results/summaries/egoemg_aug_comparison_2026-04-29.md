# EgoEMG With Augmentation — Three-Size Comparison

Evaluated 2026-04-29. All three models trained with correct EMG data augmentation
(transform bug fixed 2026-04-28).

## Overall

| | Small | Middle | Large |
|---|---|---|---|
| **Architecture** | 256d, 4 heads, 3 layers | 256d, 8 heads, 6 layers | 384d, 12 heads, 8 layers |
| **Test mean MAE** | **0.2712** | 0.2794 | 0.2795 |
| **Val MAE** | 0.2618 | 0.2629 | 0.2619 |
| **Val→Test gap** | +0.0094 | +0.0165 | +0.0176 |
| **Best epoch** | 91 | 87 | 61 |
| **Fingertip dist (mm)** | 45.0 | 45.9 | 46.1 |
| **Landmark dist (mm)** | 26.5 | 27.0 | 27.2 |

**Small is the best model.** Middle and Large are nearly identical (0.2794 vs
0.2795), hitting a performance ceiling around 0.279 on this dataset.

## Per-Split Breakdown

| Split | Small | Middle | Large | Winner |
|---|---|---|---|---|
| user/left | **0.2752** | 0.2961 | 0.2914 | Small |
| user/right | **0.2867** | 0.2991 | 0.2970 | Small |
| gesture/left | 0.2345 | **0.2199** | 0.2208 | Middle |
| gesture/right | 0.2538 | 0.2425 | **0.2420** | Large≈Middle |
| both/left | **0.2810** | 0.3112 | 0.3107 | Small |
| both/right | **0.2962** | 0.3076 | 0.3150 | Small |

Aggregated by difficulty:

| Difficulty | Small | Middle | Large | Trend |
|---|---|---|---|---|
| **gesture** (seen user + seen gesture) | 0.2442 | **0.2312** | **0.2314** | larger wins |
| **user** (new user + seen gesture) | **0.2810** | 0.2976 | 0.2942 | smaller wins |
| **both** (new user + new gesture) | **0.2886** | 0.3094 | 0.3129 | smaller wins |

Larger models win on easy samples (seen users). Smaller model generalizes
substantially better to unseen users — a classic overfitting signature.

## Left vs Right Hand

| Model | Left mean | Right mean | Δ |
|---|---|---|---|
| Small | 0.2636 | 0.2789 | right +0.015 |
| Middle | 0.2757 | 0.2831 | right +0.007 |
| Large | 0.2743 | 0.2847 | right +0.010 |

Right hand is consistently harder across all model sizes.

## Per-Joint Error (mean over 6 splits)

| Joint | Small | Middle | Large | Relative difficulty |
|---|---|---|---|---|
| Thumb | 0.2362 | 0.2430 | 0.2387 | easiest |
| Index | 0.2458 | 0.2551 | 0.2595 | |
| Proximal | 0.2538 | 0.2625 | 0.2637 | |
| Wrist flexion | 0.2776 | 0.2744 | 0.2796 | |
| Middle | 0.2878 | 0.2939 | 0.2940 | |
| Ring | 0.2835 | 0.2919 | 0.2935 | |
| Wrist deviation | 0.2012 | 0.2026 | 0.2014 | easy |
| Pinky | 0.3189 | 0.3333 | 0.3313 | hardest |
| Mid-phalanx | 0.4049 | 0.4180 | 0.4170 | hardest (MCP midpoint) |
| Distal | 0.1851 | 0.1910 | 0.1892 | easy |

Error hierarchy is highly consistent across models:
distal < wrist-dev ≈ thumb < index < proximal < wrist-flex ≈ middle ≈ ring < pinky < mid-phalanx

Distal and thumb are easiest (close to palm, limited range). Pinky and
mid-phalanx are hardest (end-effector, large kinematic freedom).

## Generalization Gap

```
Small:  val 0.2618 → test 0.2712  |  Δ = 0.009  (3.6%)
Middle: val 0.2629 → test 0.2794  |  Δ = 0.017  (6.3%)
Large:  val 0.2619 → test 0.2795  |  Δ = 0.018  (6.7%)
```

All three models have near-identical val MAE (~0.262), but the generalization
gap grows linearly with parameter count. Middle and Large overfit to the
training distribution; Small sits at the underfit/overfit balance point for
the current dataset size.

## Summary

```
         gesture       user         both        overall
Small    0.244         0.281        0.289       0.271  ← best
Middle   0.231         0.298        0.309       0.279
Large    0.231         0.294        0.313       0.279
         ↑ larger wins  ↑ smaller wins  ↑ smaller wins
```

Next steps: tune augmentation strength on the Small architecture (current
aug policy was designed for intermediate representations and may be friendlier
to smaller models), or add stronger dropout/weight decay on Middle/Large to
close the generalization gap.
