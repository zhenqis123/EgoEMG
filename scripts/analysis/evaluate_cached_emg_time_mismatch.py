"""Time-mismatch frozen EMG predictions on identical recent-object samples.

python -m scripts.analysis.evaluate_cached_emg_time_mismatch --output <directory>
Permuting deterministic eval outputs is equivalent to substituting their input
windows. No GT enters donor selection; no neural inference or GPU is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def distant_rotation(times, minimum_gap_us, rng):
    """Bijective circular permutation; reject groups lacking a legal rotation."""
    n = len(times)
    if n < 2:
        return None
    order = np.argsort(times, kind="stable")
    sorted_times = np.asarray(times)[order]
    for shift in rng.permutation(np.arange(1, n)):
        donor = np.roll(order, int(shift))
        if np.all(np.abs(np.asarray(times)[donor] - sorted_times) > minimum_gap_us):
            result = np.empty(n, np.int64)
            result[order] = donor
            return result
    return None


def mismatch_indices(sessions, objects, times, valid, minimum_gap_us=4_000_000):
    """Match donors within a session, object and hand; uncovered rows are -1."""
    sessions, objects, times, valid = map(np.asarray, (sessions, objects, times, valid))
    if objects.shape != valid.shape or objects.ndim != 2:
        raise ValueError("objects and valid must have matching (samples, hands) shapes")
    if len(sessions) != len(objects) or times.shape != (len(objects),):
        raise ValueError("sessions and timestamps must match the sample count")
    result = np.full(objects.shape, -1, dtype=np.int64)
    rng = np.random.default_rng(42)
    for hand in range(objects.shape[1]):
        groups = {}
        for row in np.flatnonzero(valid[:, hand]):
            groups.setdefault((sessions[row], objects[row, hand]), []).append(row)
        for group in groups.values():
            rows = np.asarray(group)
            donors = distant_rotation(times[rows], minimum_gap_us, rng)
            if donors is not None:
                result[rows, hand] = rows[donors]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "logs/modality_comparison_october_v5_2026-10-28/full"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.source / "emg_advantage.csv", dtype={"subject": str})
    metadata = metadata[metadata["split"] == "recent_object"].reset_index(drop=True)
    arrays = np.load(args.source / "predictions.npz")
    image = metadata.image_index.to_numpy(int)
    hand = metadata.hand_index.to_numpy(int)
    assert not metadata.duplicated(["image_index", "hand_index"]).any()
    assert arrays["available"][image, hand].all()
    e, v, y = [arrays[k][image, hand].astype(np.float64)
               for k in ("emg_pred", "vision_pred", "target")]
    valid = arrays["valid"][image, hand]
    assert np.isfinite(e).all() and np.isfinite(v).all()
    assert np.isfinite(y[valid]).all()
    times = metadata.timestamp_us.to_numpy(np.int64)
    groups = list(metadata.groupby(["session_id", "object_id", "hand_index"],
                                    sort=True).indices.values())
    counts = valid.sum(-1)

    def error(prediction):
        return np.where(valid, np.abs(prediction - y), 0).sum(-1)

    aligned = error(e)
    vision = error(v)
    train_folder = Path("logs/train_output_blend_calibration_2026-10-28/train_predictions")
    identities = np.load(train_folder / "sample_identity.npy", mmap_mode="r")
    sessions = json.loads((train_folder / "sessions.json").read_text())
    october = np.array(["202610" in s["session_id"] for s in sessions])[identities[:, 0]]
    train_y = np.load(train_folder / "target.npy", mmap_mode="r")[october].astype(float)
    train_valid = np.load(train_folder / "valid.npy", mmap_mode="r")[october]
    means = np.where(train_valid, train_y, 0).sum(0) / train_valid.sum(0)
    mean_error = error(means[hand])
    group_mean = np.empty_like(e)
    for rows in groups:
        group_mean[rows] = e[rows].mean(0)
    base = dict(
        paired_hands=len(metadata), valid_angles=int(counts.sum()),
        vision_mae=float(vision.sum() / counts.sum()),
        aligned_emg_mae=float(aligned.sum() / counts.sum()),
        october_train_mean_mae=float(mean_error.sum() / counts.sum()),
        group_mean_emg_prediction_mae=float(error(group_mean).sum() / counts.sum()),
    )
    results = {}
    subjects = metadata.subject.to_numpy(str)
    for seconds in (4, 10, 30):
        errors, masks, records = [], [], []
        for seed in range(args.repeats):
            rng = np.random.default_rng(20261029 + seed)
            partners = np.full(len(metadata), -1, np.int64)
            for rows in groups:
                donor = distant_rotation(times[rows], seconds * 1_000_000, rng)
                if donor is None:
                    continue
                partners[rows] = rows[donor]
            covered = partners >= 0
            assert np.any(covered)
            assert np.all(np.abs(times[partners[covered]] - times[covered]) > seconds * 1e6)
            for column in ("session_id", "object_id", "hand_index"):
                values = metadata[column].to_numpy()
                assert np.array_equal(values[covered], values[partners[covered]])
            assert len(np.unique(partners[covered])) == int(covered.sum())
            replacement = e.copy()
            replacement[covered] = e[partners[covered]]
            mismatch = error(replacement)
            count = counts[covered].sum()
            records.append(dict(seed=seed, coverage=float(count / counts.sum()),
                                aligned_mae=float(aligned[covered].sum() / count),
                                mismatch_mae=float(mismatch[covered].sum() / count)))
            errors.append(mismatch)
            masks.append(covered)
            if seed == 0:
                np.savez_compressed(args.output / f"partners_gap{seconds}s.npz",
                                    target_image=image, target_hand=hand,
                                    partner_row=partners, covered=covered)
        assert all(np.array_equal(masks[0], c) for c in masks)
        covered = masks[0]
        mismatch_mean = np.mean(errors, axis=0)
        scores = [r["mismatch_mae"] for r in records]
        paired_aligned = records[0]["aligned_mae"]
        results[str(seconds)] = dict(
            minimum_gap_seconds=seconds, coverage=records[0]["coverage"],
            aligned_mae=paired_aligned, mismatch_mae_mean=float(np.mean(scores)),
            mismatch_mae_std=float(np.std(scores)),
            relative_increase_percent=float(100 * (np.mean(scores) / paired_aligned - 1)),
            per_subject={}, repetitions=records,
        )
        for subject in np.unique(subjects):
            select = covered & (subjects == subject)
            count = counts[select].sum()
            if count:
                results[str(seconds)]["per_subject"][subject] = dict(
                    paired_hands=int(select.sum()),
                    aligned_mae=float(aligned[select].sum() / count),
                    mismatch_mae=float(mismatch_mean[select].sum() / count),
                )
    output = dict(
        protocol="Frozen EMG-only; entire cached recent-object paired pool; same session/object/hand bijective time mismatch",
        selection="Partners selected by metadata and timestamps only, without GT",
        output_equivalence="Deterministic eval-mode predictions with identical fixed center readout; donor input and donor cached output substitutions are equivalent",
        variability="Std is across random permutations, not a generalization confidence interval",
        repeats=args.repeats, groups=len(groups), baseline=base, results=results,
    )
    (args.output / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(base, indent=2))
    for gap, result in results.items():
        print(gap, json.dumps({k: v for k, v in result.items()
                              if k not in ("repetitions", "per_subject")}))


if __name__ == "__main__":
    main()
