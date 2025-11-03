# emg2pose_pretrain

# Precompute windows and ik failure masks

```shell
python scripts/cache_windows.py
```

# Test analysis on precomputed data(same as the original data)

```shell
python -m emg2pose.cached_test_analysis
```

# Train

```shell
python -m emg2pose.train
```

# TODO 
- [ ] CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals
- [ ] emg2tendon: From sEMG Signals to Tendon Control in Musculoskeletal Hands
- [ ] Scaling and Distilling Transformer Models for sEMG
