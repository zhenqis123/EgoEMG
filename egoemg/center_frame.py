"""Center-frame evaluation for vision/fusion pose models.

Merges the unified center-frame logic (formerly ``scripts/eval/unified_center_eval.py``)
into the test_analysis entrypoint so a single command can evaluate EMG, vision,
and fusion checkpoints on identical center frames.

Reference grid: WL=7790, stride=7790 (no overlap, no jitter). Each model is
evaluated on windows centered at those SAME frames using its own trained
window length, so supervision targets and vision frames are identical across
models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

REF_WL = 7790
repo_root = Path(__file__).resolve().parents[1]


def collect_val_centers(
    memmap_dir: Path, required_window_length: int | None = None
) -> dict[int, list[tuple[int, int]]]:
    """Collect val centers on the reference (REF_WL=7790) grid.

    Returns {episode_index: [(center, start), ...]} for centers whose split is
    user/gesture/both, optionally restricted to those that can host a full
    window of ``required_window_length`` inside their episode.
    """
    m = json.load(open(memmap_dir / "manifest.json"))
    # Resolve val split ids by name so a memmap with a different split-table
    # order is not silently mis-selected (legacy hardcode was [1, 2, 3]).
    labels = list(m.get("frame_split_labels") or [])
    val_split_ids = [labels.index(name) for name in ("user", "gesture", "both") if name in labels]
    if not val_split_ids:
        val_split_ids = [1, 2, 3]
    def _path(name: str, spec: dict) -> Path:
        # filename is implicit when a spec omits it (synthetic/legacy manifests)
        return memmap_dir / spec.get("filename", f"{name}.dat")

    ei = m["fields"]["episode_index"]
    episode_idx = np.memmap(
        _path("episode_index", ei), dtype=ei["dtype"], mode="r", shape=tuple(ei["shape"])
    )
    fs = m["fields"]["frame_split_id"]
    split = np.memmap(
        _path("frame_split_id", fs),
        dtype=fs["dtype"], mode="r", shape=tuple(fs["shape"]),
    )

    centers_per_episode: dict[int, list[tuple[int, int]]] = {}
    for e in range(int(episode_idx.max()) + 1):
        mask = episode_idx[:] == e
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        s0, e_end = int(idx[0]), int(idx[-1])
        n_windows = max(0, (e_end - s0 - REF_WL) // REF_WL + 1)
        if n_windows == 0:
            continue
        starts = s0 + np.arange(n_windows) * REF_WL
        centers = starts + REF_WL // 2
        val_mask = np.isin(split[centers], val_split_ids)  # user + gesture + both
        val_starts = starts[val_mask]
        val_centers = centers[val_mask]
        if required_window_length is not None:
            req_start = val_centers - required_window_length // 2
            fits = (req_start >= s0) & (req_start + required_window_length <= e_end)
            val_starts = val_starts[fits]
            val_centers = val_centers[fits]
        if len(val_centers):
            centers_per_episode[e] = list(
                zip(val_centers.tolist(), val_starts.tolist())
            )

    total = sum(len(v) for v in centers_per_episode.values())
    print(
        f"Collected {total} val centers across {len(centers_per_episode)} episodes "
        f"(REF_WL={REF_WL}, required_window_length={required_window_length})"
    )
    return centers_per_episode


def _load_module(cfg, ckpt_path: str):
    """Load a trained Lightning module from a checkpoint using the given config."""
    from egoemg.train import make_lightning_module

    module = make_lightning_module(cfg)
    kwargs = {
        "module_conf": cfg.module,
        "optimizer_conf": cfg.optimizer,
        "lr_scheduler_conf": cfg.lr_scheduler,
        "loss_weights": cfg.loss_weights,
        "datamodule": cfg.get("datamodule"),
        "pretrained_checkpoint": None,
        "pretrained_emg_checkpoint": None,
        "stage2_vision_checkpoint": None,
        "map_location": "cpu",
    }
    loader = module.__class__.load_from_checkpoint
    if "weights_only" in __import__("inspect").signature(loader).parameters:
        kwargs["weights_only"] = False
    return loader(ckpt_path, **kwargs)


def eval_center_frame(
    cfg,
    ckpt_path: str,
    memmap_dir: Path,
    center_window_length: int | None = None,
    hands=("left", "right"),
) -> dict:
    """Evaluate a vision/fusion checkpoint on unified center frames (both hands).

    Returns a dict keyed by hand with ``{overall_mae, n_valid, per_joint}``.
    """
    centers = collect_val_centers(memmap_dir, center_window_length)
    wl = int(cfg.datamodule.window_length)
    module = _load_module(cfg, ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device).eval()
    model = module.model

    results: dict[str, dict] = {}
    # Legacy saved configs store dataset fields under datamodule.dataset_conf
    # rather than as top-level egoemg_* aliases; fall back to the val dataset
    # entry for compatibility (mirrors unified_center_eval.py).
    dataset_template = OmegaConf.select(cfg, "datamodule.dataset_conf.val.0", default={})

    def cfg_or_dataset(key, default=None):
        value = OmegaConf.select(cfg, key, default=None)
        if value is None:
            value = OmegaConf.select(dataset_template, key, default=default)
        return value

    def resolve_path(value):
        """Resolve a possibly-relative data path against the repo root."""
        if not value:
            return value
        p = Path(value)
        if not p.is_absolute():
            p = repo_root / p
        return p

    for hand in hands:
        # Per-hand dataset so _getitem_center_supervised uses the target hand.
        ds = EgoEmgMemmapDataset(
            # Prefer the unified memmap like test_analysis._find_memmap_dir;
            # the legacy egoemg_memmap_dir alias may point at a non-unified
            # (or absent) pre-merge directory.
            memmap_dir=resolve_path(
                cfg_or_dataset(
                    "egoemg_unified_memmap_dir", default=memmap_dir
                )
            ),
            window_length=wl,
            stride=wl,
            allowed_splits=["user", "gesture", "both"],
            modalities=["emg", "joint_angles", "labels"],
            target_hand=hand,
            emg_field_preference=cfg_or_dataset(
                "egoemg_emg_field_preference", default="filtered_paper"
            ),
            emg_layout=cfg_or_dataset("egoemg_emg_layout", default="target_hand"),
            emg2pose_channel_indices=cfg_or_dataset(
                "egoemg_emg2pose_channel_indices", default=None
            ),
            channel_interpolate=bool(
                cfg_or_dataset("egoemg_channel_interpolate", default=False)
            ),
            norm_mode=cfg.datamodule.norm_mode,
            norm_stats_path=OmegaConf.select(
                cfg, "datamodule.per_dataset_norm_stats_path",
                default=OmegaConf.select(dataset_template, "norm_stats_path", default=None),
            ),
            dataset_name="egoemg",
            vision_num_frames=int(cfg_or_dataset("vision_num_frames", default=0) or 0),
            per_episode_crops_dir=resolve_path(
                cfg_or_dataset("per_episode_crops_dir", default=None)
            ),
            vision_patch_size=int(cfg_or_dataset("vision_patch_size", default=256) or 256),
            video_root=resolve_path(cfg_or_dataset("video_root", default=None)),
            allintra_root=resolve_path(cfg_or_dataset("allintra_root", default=None)),
            # Respect the config: vision-only configs set skip_emg_loading=true
            # (their models never touch EMG), which also avoids loading EMG
            # norm stats and the spurious "raw-EMG statistics" warning.
            skip_emg_loading=bool(cfg_or_dataset("skip_emg_loading", default=False)),
            center_target_only=bool(cfg_or_dataset("center_target_only", default=True)),
        )
        preds, targets = [], []
        count = 0
        first_err = None
        with torch.no_grad():
            for ep_idx, center_list in centers.items():
                for ref_center, _ in center_list:
                    model_start = ref_center - wl // 2
                    model_end = model_start + wl
                    ep_start = int(ds._episode_start_idx[ep_idx])
                    ep_end = int(ds._episode_end_idx[ep_idx])
                    if model_start < ep_start or model_end > ep_end:
                        count += 1
                        continue
                    try:
                        if ds.skip_emg_loading:
                            # Vision-only model: point-read the center frame
                            # only (no EMG I/O, no norm-stats noise). The
                            # center_supervised path below would KeyError on
                            # the pruned EMG fields.
                            sample = ds._getitem_vision_only_crops(
                                ep_idx, ref_center
                            )
                        else:
                            sample = ds._getitem_center_supervised(
                                ep_idx, model_start, model_end, ref_center
                            )
                    except Exception as exc:
                        if first_err is None:
                            first_err = f"{type(exc).__name__}: {exc}"
                        count += 1
                        continue
                    batch = {}
                    for k, v in sample.items():
                        if isinstance(v, np.ndarray):
                            batch[k] = torch.from_numpy(v).float().to(device).unsqueeze(0)
                        elif isinstance(v, torch.Tensor):
                            batch[k] = v.to(device) if v.ndim else v.to(device).unsqueeze(0)
                        else:
                            batch[k] = v
                    out = model(batch)
                    preds_ = out[0] if isinstance(out, tuple) else out
                    t = batch.get("joint_angles")
                    sample_mask = batch.get("label_valid_mask")
                    valid = (
                        bool(sample_mask.reshape(-1)[0].cpu().item())
                        if sample_mask is not None
                        else True
                    )
                    if not valid or t is None:
                        count += 1
                        continue
                    if preds_.ndim == 3:
                        preds_ = preds_[:, :, preds_.shape[-1] // 2]
                    preds.append(preds_.squeeze().cpu().numpy())
                    targets.append(t.squeeze().cpu().numpy())
                    count += 1
        if first_err:
            print(f"  First sample error: {first_err}")
        if not preds:
            results[hand] = {"overall_mae": float("nan"), "n_valid": 0}
            continue
        preds = np.array(preds)
        targets = np.array(targets)
        per_joint = np.mean(np.abs(preds - targets), axis=0)
        results[hand] = {
            "overall_mae": float(np.mean(per_joint)),
            "n_valid": len(preds),
            "per_joint": per_joint.tolist(),
        }
        print(
            f"  {hand}: MAE = {results[hand]['overall_mae']:.4f} "
            f"({results[hand]['n_valid']} samples)"
        )
    return results
