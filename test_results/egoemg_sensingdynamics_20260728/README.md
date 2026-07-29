# SensingDynamics on EgoEMG

The best validation checkpoint was selected at epoch 44 and evaluated with
`emg2pose.test_analysis` on full 10,167-sample windows. Both hands are pooled
using the same per-user test protocol as the other reimplemented EgoEMG
baselines in the paper.

| Split | MAE (rad) | MAE (deg) |
|---|---:|---:|
| Gesture | 0.2831 +/- 0.0195 | 16.2 +/- 1.1 |
| User | 0.2865 +/- 0.0060 | 16.4 +/- 0.3 |
| Both | 0.2918 +/- 0.0134 | 16.7 +/- 0.8 |
| Overall | 0.2858 | 16.4 |

Checkpoint:
`logs/20260728/sensingdynamics_egoemg_50e_lr1e-4_eta5e-6_bs320_6gpu/train/version_0/checkpoints/sensingdynamics-egoemg-epoch=044-val_mae=0.2859.ckpt`

The implementation contains 961,539 state values (approximately 1.0M
parameters, excluding no additional pose model).
