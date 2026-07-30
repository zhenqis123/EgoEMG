#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


PATTERN_AE = re.compile(r"^S(?P<subject>\d+)_A(?P<acq>\d+)_E(?P<exercise>\d+)$")
PATTERN_EA = re.compile(r"^S(?P<subject>\d+)_E(?P<exercise>\d+)_A(?P<acq>\d+)$")
SUBJECT_RE = re.compile(r"S(?P<subject>\d+)")

NINAPRO_TO_EMG2POSE = [
    "CMC1_f",  # THUMB_CMC_FE
    "CMC1_a",  # THUMB_CMC_AA
    "MCP1",  # THUMB_MCP_FE
    "IP1",  # THUMB_IP_FE
    "MCP2_a",  # INDEX_MCP_AA (may be NaN)
    "MCP2_f",  # INDEX_MCP_FE
    "PIP2",  # INDEX_PIP_FE
    "DIP2",  # INDEX_DIP_FE
    None,  # MIDDLE_MCP_AA (not available in Ninapro angles)
    "MCP3_f",  # MIDDLE_MCP_FE
    "PIP3",  # MIDDLE_PIP_FE
    "DIP3",  # MIDDLE_DIP_FE
    "MCP4_a",  # RING_MCP_AA
    "MCP4_f",  # RING_MCP_FE
    "PIP4",  # RING_PIP_FE
    "DIP4",  # RING_DIP_FE
    "MCP5_a",  # PINKY_MCP_AA
    "MCP5_f",  # PINKY_MCP_FE
    "PIP5",  # PINKY_PIP_FE
    "DIP5",  # PINKY_DIP_FE
]

WRIST_LABELS = ("WRIST_F", "WRIST_A")
EXERCISE_TO_GROUP = {1: "A", 2: "B", 3: "C", 4: "D"}


def _load_mat(path: Path) -> dict:
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}, got {arr.shape}.")
    return arr


def _parse_session_key(stem: str) -> tuple[int, int, int] | None:
    match = PATTERN_AE.match(stem)
    if match is None:
        match = PATTERN_EA.match(stem)
    if match is None:
        return None
    subject = int(match.group("subject"))
    exercise = int(match.group("exercise"))
    acq = int(match.group("acq"))
    return (subject, exercise, acq)


def _extract_subject(mat: dict, path: Path) -> str:
    for key in ("subject", "subj"):
        if key in mat:
            try:
                return str(int(np.asarray(mat[key]).reshape(())))
            except Exception:
                return str(mat[key])
    match = SUBJECT_RE.search(path.stem)
    if match:
        return match.group("subject")
    return "unknown"


def _iter_mat_files(root: Path, require_session: bool) -> list[Path]:
    files = []
    for p in root.rglob("*.mat"):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        if require_session and _parse_session_key(p.stem) is None:
            continue
        files.append(p)
    return sorted(files)


def _normalize_label(label: str) -> str:
    label = str(label).strip()
    if ":" in label:
        label = label.split(":", 1)[1].strip()
    return label


def _get_order_labels(mat: dict, num_channels: int) -> list[str]:
    order = mat.get("order_of_angles")
    if order is None:
        return [f"ch{i+1}" for i in range(num_channels)]
    order_list = np.atleast_1d(order).tolist()
    return [_normalize_label(x) for x in order_list]


def _extract_emg2pose_angles(angles: np.ndarray, order_labels: list[str]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(order_labels)}
    out = np.zeros((angles.shape[0], len(NINAPRO_TO_EMG2POSE)), dtype=np.float32)
    for out_idx, label in enumerate(NINAPRO_TO_EMG2POSE):
        if label is None:
            continue
        idx = label_to_idx.get(label)
        if idx is None:
            raise KeyError(f"Missing Ninapro angle label: {label}")
        out[:, out_idx] = angles[:, idx]
    return out


def _extract_wrist_angles(angles: np.ndarray, order_labels: list[str]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(order_labels)}
    out = np.zeros((angles.shape[0], len(WRIST_LABELS)), dtype=np.float32)
    for out_idx, label in enumerate(WRIST_LABELS):
        idx = label_to_idx.get(label)
        if idx is None:
            raise KeyError(f"Missing Ninapro wrist label: {label}")
        out[:, out_idx] = angles[:, idx]
    return out


def _build_gesture_vocab(label_map: dict) -> tuple[list[str], dict[str, int]]:
    labels = [label_map["Rest"]]
    for group in ("A", "B", "C", "D"):
        group_map = label_map.get(group)
        if not isinstance(group_map, dict):
            continue
        for key in sorted(group_map.keys(), key=lambda x: int(x)):
            labels.append(group_map[key])
    return labels, {label: idx for idx, label in enumerate(labels)}


def _map_gesture_ids(
    restimulus: np.ndarray,
    exercise: int,
    label_map: dict,
    gesture_to_id: dict[str, int],
) -> np.ndarray:
    group = EXERCISE_TO_GROUP.get(exercise)
    if group is None or group not in label_map:
        rest_id = gesture_to_id[label_map["Rest"]]
        return np.full(restimulus.shape[0], rest_id, dtype=np.int32)
    group_map = label_map[group]
    restimulus_i = restimulus.astype(int, copy=False)
    if (restimulus_i < 0).any():
        restimulus_i = restimulus_i.copy()
        restimulus_i[restimulus_i < 0] = 0
    max_rest = int(restimulus_i.max()) if restimulus_i.size else 0
    remap = np.full(max_rest + 1, -1, dtype=np.int32)
    remap[0] = gesture_to_id[label_map["Rest"]]
    for key, label in group_map.items():
        idx = int(key)
        if idx <= max_rest:
            remap[idx] = gesture_to_id[label]
    gesture_id = remap[restimulus_i]
    if (gesture_id < 0).any():
        gesture_id = gesture_id.copy()
        gesture_id[gesture_id < 0] = gesture_to_id[label_map["Rest"]]
    return gesture_id


def _raw_gesture_labels(max_id: int) -> list[str]:
    return [str(i) for i in range(max_id + 1)]


def _write_hdf5(
    out_path: Path,
    emg: np.ndarray,
    gesture_id: np.ndarray,
    gesture_labels: list[str],
    attrs: dict[str, object],
    overwrite: bool,
    joint_angles: np.ndarray | None = None,
) -> None:
    if out_path.exists():
        if not overwrite:
            print(f"Skip existing: {out_path}")
            return
        out_path.unlink()

    with h5py.File(out_path, "w") as f:
        g = f.create_group("emg2pose")
        chunk_len = min(4096, int(emg.shape[0])) or 1
        g.create_dataset(
            "emg",
            data=emg.astype(np.float32, copy=False),
            chunks=(chunk_len, emg.shape[1]),
        )
        g.create_dataset("gesture_id", data=gesture_id.astype(np.int32, copy=False))
        str_dt = h5py.string_dtype(encoding="utf-8")
        g.create_dataset(
            "gesture_labels", data=np.array(gesture_labels, dtype=object), dtype=str_dt
        )
        if joint_angles is not None:
            g.create_dataset(
                "joint_angles",
                data=joint_angles.astype(np.float32, copy=False),
                chunks=(chunk_len, joint_angles.shape[1]),
            )
        g.attrs.update(attrs)


def _pick_angle_root(db: str, base: Path) -> Path | None:
    old = base / "DB9_Old_OrderedByManfredo2023" / db
    direct = base / db
    if old.exists():
        return old
    if direct.exists():
        return direct
    return None


def _collect_subjects(files: list[Path]) -> list[int]:
    subjects: set[int] = set()
    for p in files:
        key = _parse_session_key(p.stem)
        if key is None:
            continue
        subjects.add(key[0])
    return sorted(subjects)


def _convert_db(
    db: str,
    emg_root: Path,
    angle_root: Path | None,
    out_root: Path,
    label_map: dict,
    label_mode: str,
    overwrite: bool,
) -> None:
    out_db = out_root / db
    out_db.mkdir(parents=True, exist_ok=True)

    require_session = angle_root is not None
    emg_files = _iter_mat_files(emg_root, require_session=require_session)
    emg_map: dict[tuple[int, int, int], Path] = {}
    for p in emg_files:
        key = _parse_session_key(p.stem)
        if key is not None and key not in emg_map:
            emg_map[key] = p

    angle_files: list[Path] = []
    subject_map: dict[int, int] = {}
    if angle_root is not None:
        angle_files = _iter_mat_files(angle_root, require_session=True)
        angle_subjects = _collect_subjects(angle_files)
        emg_subjects = _collect_subjects(emg_files)
        if angle_subjects and emg_subjects and len(angle_subjects) == len(emg_subjects):
            subject_map = dict(zip(angle_subjects, emg_subjects))

    gesture_labels, gesture_to_id = _build_gesture_vocab(label_map)

    if angle_root is None:
        for p in emg_files:
            mat = _load_mat(p)
            if "emg" not in mat:
                print(f"Skip (no emg): {p}")
                continue
            emg = _as_2d(mat["emg"], "emg")
            restimulus = np.asarray(mat.get("restimulus", [])).squeeze()
            if restimulus.ndim == 0 or restimulus.size == 0:
                print(f"Skip (no restimulus): {p}")
                continue
            exercise = int(np.asarray(mat.get("exercise", -1)).reshape(()))
            if label_mode == "mapped":
                gesture_id = _map_gesture_ids(restimulus, exercise, label_map, gesture_to_id)
                labels = gesture_labels
            else:
                rest = restimulus.astype(int, copy=False)
                if (rest < 0).any():
                    rest = rest.copy()
                    rest[rest < 0] = 0
                gesture_id = rest
                labels = _raw_gesture_labels(int(rest.max()) if rest.size else 0)

            length = min(emg.shape[0], gesture_id.shape[0])
            emg = emg[:length]
            gesture_id = gesture_id[:length]

            session = p.stem
            user = _extract_subject(mat, p)
            attrs = {
                "dataset": f"Ninapro_{db}",
                "session": session,
                "user": user,
                "num_channels": int(emg.shape[1]),
                "exercise": int(exercise),
            }
            out_path = out_db / f"{session}.hdf5"
            _write_hdf5(
                out_path=out_path,
                emg=emg,
                gesture_id=gesture_id,
                gesture_labels=labels,
                attrs=attrs,
                overwrite=overwrite,
            )
            print(f"Wrote {out_path}")
        return

    for angle_path in angle_files:
        key = _parse_session_key(angle_path.stem)
        if key is None:
            continue
        subject, exercise, acq = key
        emg_subject = subject_map.get(subject, subject)
        emg_key = (emg_subject, exercise, acq)
        emg_path = emg_map.get(emg_key)
        if emg_path is None:
            print(f"Skip (missing emg): {angle_path}")
            continue

        angle_mat = _load_mat(angle_path)
        if "angles" not in angle_mat and "joint_angles" not in angle_mat:
            print(f"Skip (no joint_angles): {angle_path}")
            continue
        angles = angle_mat["angles"] if "angles" in angle_mat else angle_mat["joint_angles"]
        angles = _as_2d(angles, "angles")
        order_labels = _get_order_labels(angle_mat, angles.shape[1])
        try:
            emg2pose_angles = _extract_emg2pose_angles(angles, order_labels)
            wrist_angles = _extract_wrist_angles(angles, order_labels)
            joint_angles = np.concatenate([emg2pose_angles, wrist_angles], axis=1)
        except KeyError as exc:
            print(f"Skip angles ({angle_path}): {exc}")
            joint_angles = None

        emg_mat = _load_mat(emg_path)
        if "emg" not in emg_mat:
            print(f"Skip (no emg): {emg_path}")
            continue
        emg = _as_2d(emg_mat["emg"], "emg")

        restimulus = np.asarray(emg_mat.get("restimulus", [])).squeeze()
        if restimulus.ndim == 0 or restimulus.size == 0:
            restimulus = np.asarray(angle_mat.get("restimulus", [])).squeeze()
        if restimulus.ndim == 0 or restimulus.size == 0:
            print(f"Skip (no restimulus): {angle_path}")
            continue

        if label_mode == "mapped":
            gesture_id = _map_gesture_ids(restimulus, exercise, label_map, gesture_to_id)
            labels = gesture_labels
        else:
            rest = restimulus.astype(int, copy=False)
            if (rest < 0).any():
                rest = rest.copy()
                rest[rest < 0] = 0
            gesture_id = rest
            labels = _raw_gesture_labels(int(rest.max()) if rest.size else 0)

        length = emg.shape[0]
        if joint_angles is not None:
            length = min(length, joint_angles.shape[0])
            joint_angles = joint_angles[:length]
        length = min(length, gesture_id.shape[0])
        emg = emg[:length]
        gesture_id = gesture_id[:length]

        session = angle_path.stem
        attrs = {
            "dataset": f"Ninapro_{db}",
            "session": session,
            "user": str(emg_subject),
            "num_channels": int(emg.shape[1]),
            "exercise": int(exercise),
        }
        if joint_angles is not None:
            attrs["num_joint_angles"] = int(joint_angles.shape[1])
        out_path = out_db / f"{session}.hdf5"
        _write_hdf5(
            out_path=out_path,
            emg=emg,
            gesture_id=gesture_id,
            gesture_labels=labels,
            attrs=attrs,
            overwrite=overwrite,
            joint_angles=joint_angles,
        )
        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Ninapro DB1-DB8 into unified HDF5 datasets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/emg_corpus/Ninapro"),
        help="Ninapro root directory.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/emg_corpus/Ninapro_relabeled"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("assets/DB_label_map.json"),
        help="Path to DB label map JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .hdf5 files.",
    )
    args = parser.parse_args()

    with args.label_map.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    ninapro_root = args.root
    angle_base = ninapro_root / "DB9"

    db_configs = {
        "DB1": {
            "emg_root": ninapro_root / "DB1",
            "angle_root": _pick_angle_root("DB1", angle_base),
            "label_mode": "mapped",
        },
        "DB2": {
            "emg_root": ninapro_root / "DB2_Preproc",
            "angle_root": _pick_angle_root("DB2", angle_base),
            "label_mode": "mapped",
        },
        "DB3": {
            "emg_root": ninapro_root / "db3_Preproc",
            "angle_root": None,
            "label_mode": "raw",
        },
        "DB4": {
            "emg_root": ninapro_root / "DB4_Preproc",
            "angle_root": None,
            "label_mode": "raw",
        },
        "DB5": {
            "emg_root": ninapro_root / "DB5_Preproc",
            "angle_root": _pick_angle_root("DB5", angle_base),
            "label_mode": "mapped",
        },
        "DB6": {
            "emg_root": ninapro_root / "DB6_Preproc",
            "angle_root": None,
            "label_mode": "raw",
        },
        "DB7": {
            "emg_root": ninapro_root / "DB7_Preproc",
            "angle_root": None,
            "label_mode": "raw",
        },
        "DB8": {
            "emg_root": ninapro_root / "DB8",
            "angle_root": None,
            "label_mode": "raw",
        },
    }

    for db, cfg in db_configs.items():
        if not cfg["emg_root"].exists():
            print(f"Skip {db}: missing {cfg['emg_root']}")
            continue
        _convert_db(
            db=db,
            emg_root=cfg["emg_root"],
            angle_root=cfg["angle_root"],
            out_root=args.out_root,
            label_map=label_map,
            label_mode=cfg["label_mode"],
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
