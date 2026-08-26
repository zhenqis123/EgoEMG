# EgoEMG data — known issues

Check this file before assuming a particular field's value is real. The labelled
fields are supervised targets; the ones below carry caveats about what they
actually contain.

## `generated_mano_*_pose` is not a reliable supervised signal

`generated_mano_left_pose` / `generated_mano_right_pose` are `float32 (N, 48)`
MANO pose (not full hand shape) labels. They are:

1. **Zero-filled** on rows where `generated_label_valid = false`. In the full
   dataset ~32% of rows have `generated_label_valid = false`, so a large share
   of the MANO pose rows read as the constant zero pose.
2. **Static plateaus** on many of the remaining (valid) rows — within a long
   stretch the value does not change, so the row does not vary across the time
   window.

Combined, `generated_mano_*_pose` is constant for roughly a third of windows in
a small dataset preview. **This is why replaying the ground-truth MANO meshes
does not track the gestures in the reference video** (GitHub issue #3): the
mesh is replaying a zero-filled / static pose, not the recording.

### What to replay instead

- Replay the **supervised target**: `generated_joint_angles_left` /
  `generated_joint_angles_right` (20 finger joint angles) plus the wrist
  `mocap_*_wrist_pitch` / `mocap_*_wrist_yaw` — the 22-dim target the EMG
  model is trained on.
- You can also replay the **mano betas** (`generated_mano_*_beta`, per-recording
  shape) with `mocap_mano_*_world_transform` to place the hand in world space.
- Only compare **valid rows**: filter on `generated_label_valid` (both hands)
  and, in a preview shard, the `is_first`/`is_last` recording boundaries.
- For a faithful replay, use FK over the joint angles rather than the MANO pose
  field. The repo's `visualize_dataset.py` `fk_vs_mano` mode compares the two.

## IMU channel order

The published `dataset_egoemg_unified` carries the corrected IMU layout
`[acc, gyro]`. Any copy saved before 2026-08-20 stored the old order and can be
repaired in place with
`scripts/prepare/fix_egoemg_imu_channel_order.py`. In the share this is why a
`_imu_fix_20260820/` backup directory existed; it has been moved out of the
public share. The IMU values on **placeholder** IMU recordings should not be
treated as time-synchronised ground truth.

## Filtered EMG is published precomputed

`emg_*_filtered_paper` is served as a precomputed filter variant; the
reproducible filter pipeline is not released, so treat the values as given and
match them with `emg_field_preference=filtered_paper` (the v3 default).
The legacy `filtered` preference is a v2 name that no longer resolves in v3.

## ShowEE / Incre rows in the merged dataset

The unified memmap merges EgoEMG, ShowEE, and Incre (see the dataset-merge
section of `CLAUDE.md`):

- **ShowEE** (`dataset_source_id=1`): wrist angles zero-filled,
  `wrist_angles_valid=false`; the wrist loss is masked. Validation always uses
  EgoEMG-only rows.
- **Incre** (`dataset_source_id=2`): mocap/vision unavailable (stale flags),
  `mocap_*_valid` all false, `label_gesture_active` false with class 0. Only
  the right-hand EMG + finger joint angles are supervised.

## Vision and video caveats

- Frame reads require **all-intra** videos and `decord`; there is no OpenCV /
  original-video fallback.
- Precomputed crops only exist for frames with ≥2 in-view valid markers, so a
  few episode-head frames (notably frame 0) have no crop keys and are skipped.
- Some released all-intra videos are anomalously small relative to their
  episode length (e.g. `episode_000004` ~15 MB, `episode_000030` ~280 MB),
  which indicates an incomplete/short recording — visualise with care.

## Layout note

The published `dataset_egoemg_unified` and the preview shard use a **flat**
memmap layout (`manifest.json` filenames like `timestamp_us.dat`, no group
subdirectories). The reader resolves files directly from the manifest, so this
is a drop-in for training / evaluation; it is not required to have
`modality_groups` in the manifest (the validator only cross-checks it when
present).
