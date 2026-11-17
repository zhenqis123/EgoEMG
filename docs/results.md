# Benchmark results

See the [repository README](../README.md#eval) for checkpoint evaluation commands.
The tables include comparisons beyond the released models; consult the
[checkpoint inventory](workflows.md#released-checkpoint-bundle) for available weights.
Bold values mark the lowest error in each metric column within a table.
Published EMG2Pose baseline results are described in the
[official benchmark](https://github.com/facebookresearch/emg2pose).

## EgoEMG benchmark

The following tables reproduce the reported benchmark results. Values are
joint-angle **MAE in degrees (↓)** over 22 finger and wrist angles. User, Gesture,
User+Gesture, and Object evaluate generalization to unseen users, gestures, both, and
objects, respectively. Overall pools valid angles across the four conditions
and both hands.

**EMG-only models.** Predictions are evaluated throughout each EMG window.

| Method | Params | Gesture | User | User+Gesture | Object | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| EMGFormer-S | 3.5M | **13.40** | 15.37 | 16.87 | **9.92** | **12.28** |
| EMGFormer-M | 6.6M | **13.40** | 15.69 | 17.19 | 10.15 | 12.46 |
| EMGFormer-L | 16.3M | 13.92 | **14.83** | **16.14** | 10.29 | 12.46 |
| NeuroPose | 6.4M | 16.25 | 15.79 | 16.62 | 12.63 | 14.45 |
| emg2pose | 3.0M | 14.19 | 16.06 | 17.04 | 10.16 | 12.76 |
| SensingDynamics | 1.0M | 15.82 | 16.11 | 16.24 | 11.39 | 13.43 |
| vemg2pose | 6.0M | 14.14 | 16.67 | 17.69 | 10.37 | 13.00 |

**Single-frame vision and residual fusion.** Vision and fusion are evaluated
on the same center frames. Δ is the reduction in Overall MAE from adding EMG,
computed before rounding.

| Backbone | Input | Gesture | User | User+Gesture | Object | Overall | MAE reduction (°) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/14 | Vision | 7.68 | 8.74 | 9.09 | 9.02 | 8.55 | -- |
| ViT-S/14 | + EMG | 7.36 | 8.33 | 8.47 | 8.34 | 8.05 | +0.51 |
| ViT-B/14 | Vision | 7.15 | 8.13 | 8.33 | 8.55 | 8.02 | -- |
| ViT-B/14 | + EMG | 6.87 | 7.85 | 7.96 | 8.07 | 7.65 | +0.37 |
| ViT-L/14 | Vision | 6.55 | 7.69 | 7.67 | 7.81 | 7.39 | -- |
| ViT-L/14 | + EMG | **6.23** | **7.48** | **7.32** | **7.41** | **7.06** | +0.33 |

**Fusion design and temporal context (ViT-S/14).** Temporal models use 17 RGB
frames over four seconds; all rows are evaluated at the center frame.

| Model | Gesture | User | User+Gesture | Object | Overall |
| --- | --- | --- | --- | --- | --- |
| ViT-S | 7.68 | 8.74 | 9.09 | 9.02 | 8.55 |
| Direct fusion | 7.39 | 8.55 | 8.61 | 8.40 | 8.14 |
| Residual fusion | 7.36 | 8.33 | 8.47 | 8.34 | 8.05 |
| Temporal | 6.66 | 7.89 | 7.83 | 8.37 | 7.72 |
| Temporal + EMG | **6.51** | **7.74** | **7.68** | **7.99** | **7.47** |

<details>
<summary>ResNet vision and residual-fusion results</summary>

| Backbone | Input | Gesture | User | User+Gesture | Object | Overall | MAE reduction (°) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 | Vision | 9.31 | 10.33 | 10.29 | 10.24 | 9.98 | -- |
| ResNet-18 | + EMG | 9.05 | 10.31 | 10.27 | 9.76 | 9.71 | +0.27 |
| ResNet-50 | Vision | 9.06 | 10.31 | 10.41 | 10.12 | 9.85 | -- |
| ResNet-50 | + EMG | 8.68 | 9.99 | 9.99 | 9.31 | 9.30 | +0.55 |
| ResNet-152 | Vision | 7.94 | 8.94 | 8.98 | 9.21 | 8.76 | -- |
| ResNet-152 | + EMG | **7.75** | **8.84** | **8.80** | **8.70** | **8.45** | +0.31 |

Δ is the reduction in Overall MAE from adding EMG, computed before rounding.

</details>

## EMG2Pose benchmark

The following table reproduces the EMG2Pose benchmark comparison.
Values are **MAE in degrees, mean ± SD across users**. EMGFormer follows the
official User, Stage, and User+Stage test splits, using non-overlapping
7,790-sample windows with noncausal context and population SD across users.
Entries marked † are previously reported results and retain their original
evaluation protocols, including causal evaluation for Position and Velocity MT.
MT denotes joint Regression+Tracking training, evaluated in Regression mode.
NeuroPose and SensingDynamics values are taken from the EMG2Pose benchmark.

| Method | Params | User | Stage | User+Stage |
| --- | --- | --- | --- | --- |
| EMGFormer-S | 3.5M | 11.4 ± 1.1 | **12.9 ± 1.5** | 14.1 ± 1.2 |
| EMGFormer-M | 6.6M | 11.3 ± 1.0 | 13.6 ± 1.6 | 14.1 ± 1.1 |
| EMGFormer-L | 16.3M | **11.2 ± 1.1** | 13.2 ± 1.6 | **13.9 ± 1.2** |
| CLDM† | 7.0M | 11.3 ± 1.0 | 14.3 ± 1.5 | 14.7 ± 1.4 |
| Position MT† | 6.0M | 11.5 ± 1.2 | 14.0 ± 1.6 | 14.6 ± 1.3 |
| Velocity MT† | 6.0M | 11.6 ± 1.3 | 13.9 ± 1.6 | 14.6 ± 1.3 |
| vemg2pose† | 6.0M | 12.2 ± 1.3 | 15.2 ± 1.6 | 15.8 ± 1.4 |
| emg2pose (position)† | 3.0M | 12.6 ± 1.3 | 15.2 ± 1.6 | 15.6 ± 1.3 |
| NeuroPose† | 6.4M | 13.2 ± 1.1 | 17.2 ± 1.7 | 17.5 ± 1.5 |
| SensingDynamics† | 1.0M | 15.5 ± 1.4 | 18.8 ± 1.6 | 18.7 ± 1.6 |
