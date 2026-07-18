# Optuna Experiment Archive

This is the single entry point for Optuna search results in this workspace.
Raw SQLite databases still live in `assets/`, but this directory contains
relative links to them plus the cleaned summaries and selected per-study
reports.

## Directory Layout

```text
ablation_study/optuna_search_archive/
├── README.md
├── databases/                  # symlinks to raw SQLite DBs in assets/
├── results/                    # cross-study summaries and top trials
└── studies/                    # optional detailed per-study reports
```

## Start Here

| Need | File |
|---|---|
| All studies, trial counts, best objective values | `results/study_summary.csv` |
| Compact machine-readable top trials | `results/top_trials.json` |
| Top trials for one study | `results/top_<study>.csv` |
| Raw Optuna DBs | `databases/*.db` |

## Study Map

| Study | DB link | Main summary | Notes |
|---|---|---|---|
| `egoemg-aug-v5` | `databases/optuna_augmentation.db` | `results/top_egoemg-aug-v5.csv` | Historical 16-channel augmentation search. Best trial: 81, `val_mae=0.250691`. Useful reference for older EgoEMG augmentation behavior. |
| `egoemg-user-mae-v2` | `databases/optuna_user_mae.db` | `results/top_egoemg-user-mae-v2.csv` | Earlier user-split objective search. |
| `egoemg-window-v4` | `databases/optuna_window_length.db` | `results/top_egoemg-window-v4.csv` | Window-length search. Detailed ablation artifacts are in `ablation_study/window_length/`. |
| `emg2pose_search` | `databases/optuna.db` | `results/top_emg2pose_search.csv` | Older EMG2Pose hyperparameter search. |

Superseded or incomplete studies are still listed in
`results/study_summary.csv` when they exist in the raw DBs, but they should not
be treated as primary evidence unless there is a specific reason.

## Paper-Use Guidance

- For the information-theory course paper, use the historical 16-channel
  augmentation search `egoemg-aug-v5` and the window-length ablation.
- The target-hand WL12000 augmentation Optuna sweeps were removed because they
  used `filtered_paper` with the raw-like fallback norm. Rerun them with the
  field-aware norm fix before using any target-hand WL12000 augmentation
  conclusions.
- Keep `val_user_mae` and aggregate `val_mae` conclusions separate when future
  sweeps are rerun: their best augmentation regions may differ.

## Raw Logs

`logs` in the repository is a symlink to `/mnt/nvme/xiziheng/logs`. During
cleanup, bulky stale and non-top trial directories were removed. The reliable
records are:

- the SQLite DBs linked under `databases/`;
- the cross-study summaries under `results/`;
- any remaining top trial directories referenced by the CSV files.

## Cleanup Policy

Keep this archive, the raw SQLite databases, and selected top trial logs. Delete
stale duplicate logs, failed/incomplete trial directories, and superseded manual
grids when disk space matters.
