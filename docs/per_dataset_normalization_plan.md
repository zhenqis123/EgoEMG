# Per-Dataset Normalization Implementation Plan

## Overview

Current batch normalization has issues when mixing datasets with vastly different scales (e.g., pimforce is 500x smaller than emg2pose). This plan implements per-dataset normalization to address this.

## Design

### 1. Pre-compute Normalization Statistics

Create a statistics file that stores per-dataset mean and std:

**File**: `assets/per_dataset_norm_stats.json`

```json
{
  "emg2pose": {"mean": 0.0, "std": 18.49},
  "pimforce": {"mean": 0.004, "std": 0.04},
  "ninapro_db1": {"mean": 0.17, "std": 0.36},
  "ninapro_db2": {"mean": 0.0, "std": 0.0001},
  "ninapro_db5": {"mean": -0.95, "std": 10.14},
  "emg2qwerty_left": {"mean": 0.0, "std": 10.56},
  "emg2qwerty_right": {"mean": 0.0, "std": 12.38}
}
```

### 2. Add `norm_mode: per-dataset` Support

**File**: `emg2pose/lightning_pretrain.py`

Add `_apply_per_dataset_norm` method:

```python
def _apply_per_dataset_norm(self, batch: dict[str, torch.Tensor]) -> None:
    """Normalize EMG using per-dataset statistics."""
    emg = batch["emg"]
    dataset_names = batch["dataset_name"]  # (B,) tensor of strings or list
    
    # Load stats (cached)
    if not hasattr(self, '_per_dataset_stats'):
        import json
        stats_path = self.hparams.datamodule.get(
            "per_dataset_norm_stats_path",
            "assets/per_dataset_norm_stats.json"
        )
        with open(stats_path) as f:
            self._per_dataset_stats = json.load(f)
    
    # Normalize each sample by its dataset
    normalized = torch.zeros_like(emg)
    for i, name in enumerate(dataset_names):
        stats = self._per_dataset_stats.get(name, {"mean": 0.0, "std": 1.0})
        mean = stats["mean"]
        std = stats["std"]
        normalized[i] = (emg[i] - mean) / (std + 1e-6)
    
    batch["emg"] = normalized
```

Update `_apply_batch_norm` to dispatch:

```python
def _apply_batch_norm(self, batch: dict[str, torch.Tensor]) -> None:
    norm_mode = self.hparams.datamodule.get("norm_mode") if self.hparams.datamodule else None
    
    if norm_mode == "batch":
        emg = batch["emg"]
        mean = emg.mean()
        std = emg.std()
        batch["emg"] = (emg - mean) / (std + 1e-6)
    elif norm_mode == "per-dataset":
        self._apply_per_dataset_norm(batch)
```

### 3. Update Config

**File**: `config/datamodule/default.yaml`

```yaml
norm_mode: null  # null | batch | instance | per-dataset
per_dataset_norm_stats_path: assets/per_dataset_norm_stats.json
```

### 4. Create Statistics Generation Script

**File**: `emg2pose/scripts/generate_per_dataset_norm_stats.py`

```python
"""Generate per-dataset normalization statistics."""

import json
import numpy as np
from pathlib import Path

# Statistics computed from emg_stats_comparison.md
STATS = {
    "emg2pose": {"mean": 0.00003, "std": 18.49},
    "pimforce": {"mean": 0.00369, "std": 0.0392},
    "ninapro_db1": {"mean": 0.165, "std": 0.361},
    "ninapro_db2": {"mean": 0.0, "std": 0.0001},  # Suspect data
    "ninapro_db5": {"mean": -0.946, "std": 10.14},
    "emg2qwerty_left": {"mean": 0.00001, "std": 10.56},
    "emg2qwerty_right": {"mean": 0.00008, "std": 12.38},
}

def main():
    out_path = Path(__file__).parent.parent.parent / "assets" / "per_dataset_norm_stats.json"
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(STATS, f, indent=2)
    
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
```

### 5. Ensure `dataset_name` is Propagated in DataLoader

The `PretrainWrapperDataset` already adds `dataset_name` to each sample. Need to ensure the collate function preserves it.

**File**: `emg2pose/datamodule.py` or custom collate

```python
def collate_fn(batch):
    """Custom collate that preserves dataset_name as list."""
    emg = torch.stack([s["emg"] for s in batch])
    # ... other fields
    dataset_names = [s["dataset_name"] for s in batch]
    return {
        "emg": emg,
        # ... other fields
        "dataset_name": dataset_names,
    }
```

## Usage

```yaml
# config/experiment/emgformer/supervised_pretrain_angle.yaml
datamodule:
  norm_mode: per-dataset
  per_dataset_norm_stats_path: assets/per_dataset_norm_stats.json
```

## Notes

1. **ninapro_db2** has near-zero values — may need investigation or exclusion
2. **emg2pose** has extreme outliers — could use robust stats (P1-P99 clipping before computing mean/std)
3. Statistics are computed from first 5M samples of each dataset