"""Load a calibration from an index entry and its optional components."""

from copy import deepcopy
import json
from pathlib import Path


FORMAT = "egoemg_calibration_split_v1"


def coordinate_metadata(value):
    """Keep coordinate identities when removing unrelated source provenance."""
    result = {}
    for key, item in value.items():
        if key == "registry_sources":
            continue
        if any(term in key.lower() for term in
               ("frame_definition", "frozen_marker", "template_points")):
            result[key] = deepcopy(item)
        elif isinstance(item, dict):
            nested = coordinate_metadata(item)
            if nested:
                result[key] = nested
        elif isinstance(item, list) and frame_contracts(item):
            result[key] = deepcopy(item)
    return result


def frame_contracts(value, prefix=""):
    """Keep frozen coordinate identities in the extrinsics deduplication key."""
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "registry_sources":
                continue
            path = prefix + "/" + key
            if any(
                term in key.lower()
                for term in ("frame_definition", "frozen_marker", "template_points")
            ):
                found[path] = item
            else:
                found.update(frame_contracts(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(frame_contracts(item, prefix + "/" + str(index)))
    return found


def assemble_calibration(record, root):
    """Resolve a split registry record, keeping provenance and frame definitions."""
    if not isinstance(record, dict) or record.get("storage_format") != FORMAT:
        return record
    root = Path(root).resolve()

    def component(name):
        reference = record.get(name)
        if reference is None:
            return None
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Calibration reference escapes dataset root: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["device"] != record["device"]:
            raise ValueError(f"Calibration component device mismatch: {path}")
        return value

    intrinsics = component("intrinsics")
    if intrinsics is None:
        raise ValueError("A split calibration requires an intrinsics reference")
    extrinsics = component("extrinsics")
    sensors = component("sensors")
    result = deepcopy(record["metadata"])
    result.update(
        calibration_id=record["calibration_id"],
        device=record["device"],
        cameras=intrinsics["cameras"],
        transforms=extrinsics["transforms"] if extrinsics else {},
        imu=sensors["imu"] if sensors else None,
        camera_imu_time_offsets=sensors["camera_imu_time_offsets"] if sensors else None,
    )
    if extrinsics and extrinsics["frame_contracts"] != frame_contracts(result):
        raise ValueError("Extrinsics and registry coordinate contracts differ")
    return result


def read_json(path):
    """Read ordinary JSON or assemble a calibrations/registry/<id>.json file."""
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and value.get("storage_format") == FORMAT:
        if path.parent.name != "registry" or path.parent.parent.name != "calibrations":
            raise ValueError(f"Unexpected split calibration registry location: {path}")
        return assemble_calibration(value, path.parents[2])
    return value


def load_calibration(root, entry):
    """Resolve a calibration index entry in either current or older layouts."""
    root = Path(root)
    if entry.get("storage_format") == FORMAT:
        return assemble_calibration(entry, root)
    if "path" in entry:
        return read_json(root / entry["path"])
    return entry
