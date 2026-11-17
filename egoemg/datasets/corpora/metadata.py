"""Corpus split metadata and label validity helpers."""

import numpy as np


def load_splits(
    metadata_file: str,
    subsample: float = 1.0,
    random_seed: int = 0,
) -> dict[str, list[str]]:
    """Load train, val, and test datasets from metadata csv."""

    import pandas as pd

    # Load dataframe
    df = pd.read_csv(metadata_file)

    # Optionally subsample
    df = df.groupby("split").apply(
        lambda x: x.sample(frac=subsample, random_state=random_seed)
    )
    df.reset_index(drop=True, inplace=True)

    # Format as dictionary
    splits = {}
    for split, df_ in df.groupby("split"):
        splits[split] = list(df_.filename)

    return splits


def get_ik_failures_mask(joint_angles: np.ndarray) -> np.ndarray:
    """Compute mask that is True where there are no ik failures."""
    zeros = np.zeros_like(joint_angles)  # (..., joint)
    is_zero = np.isclose(joint_angles, zeros)
    return ~np.all(is_zero, axis=-1)
