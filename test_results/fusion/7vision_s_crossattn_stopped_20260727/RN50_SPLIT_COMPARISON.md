# RN50-S residual fusion vs RN50 vision-only by split

Unified center-frame evaluation on identical samples. MAE is converted from
radians to degrees; lower is better.

| Split | Vision-only | Residual fusion | Improvement | Relative improvement | Hand samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| user | 6.2372 deg | 6.2139 deg | 0.0233 deg | 0.3731% | 1619 |
| gesture | 4.4415 deg | 4.2966 deg | 0.1448 deg | 3.2610% | 2222 |
| both | 6.1960 deg | 6.1568 deg | 0.0391 deg | 0.6317% | 313 |

Per-hand relative improvements:

| Split | Left | Right |
| --- | ---: | ---: |
| user | 0.6076% | 0.1128% |
| gesture | 3.3531% | 3.1649% |
| both | 0.7021% | 0.5553% |

The gain is concentrated in the gesture-held-out split. The user-held-out and
both-held-out splits remain positive, but the improvements are small.
