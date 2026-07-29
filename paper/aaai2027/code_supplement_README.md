# Anonymous Code and Data Supplement

This archive contains the implementation, Hydra configurations, normalization
statistics, and a small numeric reviewer sample for the EgoEMG hand-pose
benchmark. It is self-contained and does not rely on web-hosted supplementary
material.

## Contents

- `emg2pose/`: dataset loaders and the EMGFormer, vision, and fusion models.
- `config/`: the layered Hydra configuration tree used by the experiments.
- `config/experiment/fusion/`: vision-only baselines and the frozen/fine-tuned
  fusion configurations for all seven visual backbones in the paper.
- `scripts/eval/`: unified center-frame and per-gesture evaluation utilities.
- `scripts/data/`: the EgoEMG vision-index builder.
- `scripts/experiments/`: representative launch recipes for the fusion sweep.
- `assets/`: normalization statistics referenced by the configurations.
- `data/egoemg_reviewer_sample/`: a 16,000-frame bilateral 22-DoF
  EMG-to-pose sample.

The reviewer sample contains bilateral raw and filtered EMG, 20 finger-angle
targets plus wrist pitch/yaw for both hands, wrist-angle validity, pose-label
validity, and gesture labels for one anonymized episode. It verifies the full
22-DoF memmap loading path for either target hand; it is not intended to
reproduce full-dataset metrics. Full-scale vision and fusion experiments also
require the EgoEMG all-intra videos, precrop metadata, MANO assets, and the
initialization checkpoints described in the separately submitted supplementary
material. Their locations are represented by portable relative paths in the
packaged configurations. The supplementary PDF is intentionally not duplicated
inside this archive.

## Setup

```bash
conda env create -f environment.yml
conda activate emg2pose
pip install -e .
```

## Verify the reviewer sample

```bash
python - <<'PY'
from pathlib import Path
from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

for hand in ("left", "right"):
    dataset = EgoEmgMemmapDataset(
        memmap_dir=Path("data/egoemg_reviewer_sample"),
        window_length=12000,
        stride=12000,
        allowed_splits=["train"],
        modalities=["emg", "joint_angles", "labels"],
        target_hand=hand,
        emg_field_preference="filtered",
        emg_layout="target_hand",
    )
    sample = dataset[0]
    print(hand, len(dataset), sample["emg"].shape,
          sample["joint_angles"].shape)
PY
```

For both hands, the expected sample shapes are `(8, 12000)` for EMG and
`(22, 12000)` for the 20 finger joint angles plus two wrist articulation
angles.

## Configuration dry-run

Hydra configurations can be composed without accessing the full dataset:

```bash
python -m emg2pose.train \
  experiment=fusion/fusion_rn18_s_simple_frozen_augbest_30e \
  --cfg job
```

The complete fusion freeze ablation uses the following suffixes for each of
`rn18`, `rn50`, `rn152`, `vits`, `vitb`, `vitl`, and `wilor`:

- `fusion_<backbone>_s_simple_frozen_augbest_30e`
- `fusion_<backbone>_s_simple_unfrozen_augbest_30e`

## License

The baseline code is released under the MIT License. Third-party components
retain their original licenses. The included reviewer sample is provided only
for anonymous peer-review evaluation and must not be redistributed.
