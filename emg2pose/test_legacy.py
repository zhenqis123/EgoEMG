"""Evaluate legacy emg2pose checkpoints (StatePoseModule, VEMG2Pose, PoseModule) using memmap data."""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from tqdm import tqdm

from emg2pose.datasets.emg2pose_dataset import Emg2PoseDataset
from emg2pose.transforms import ExtractToTensor, Compose


# Map old _target_ paths to current locations
TARGET_REMAP = {
    "emg2pose.pose_modules.StatePoseModule": "emg2pose.models.modules.pose.StatePoseModule",
    "emg2pose.pose_modules.VEMG2PoseWithInitialState": "emg2pose.models.modules.pose.VEMG2PoseWithInitialState",
    "emg2pose.pose_modules.PoseModule": "emg2pose.models.modules.pose.PoseModule",
    "emg2pose.networks.TdsNetwork": "emg2pose.models.featurizers.tds.TdsNetwork",
    "emg2pose.networks.Conv1dBlock": "emg2pose.models.featurizers.tds.Conv1dBlock",
    "emg2pose.networks.TdsStage": "emg2pose.models.featurizers.tds.TdsStage",
    "emg2pose.networks.MLP": "emg2pose.models.decoders.mlp.MLP",
    "emg2pose.networks.SequentialLSTM": "emg2pose.models.decoders.lstm.SequentialLSTM",
    "emg2pose.networks.NeuroPose": "emg2pose.models.featurizers.neuropose.NeuroPose",
    "emg2pose.networks.DecoderBlock": "emg2pose.models.featurizers.neuropose.DecoderBlock",
    "emg2pose.networks.EncoderBlock": "emg2pose.models.featurizers.neuropose.EncoderBlock",
    "emg2pose.networks.ResidualBlock": "emg2pose.models.featurizers.neuropose.ResidualBlock",
}


def remap_targets(conf):
    """Recursively remap _target_ paths in a config dict."""
    from omegaconf import OmegaConf
    # Convert OmegaConf to plain dict first
    if OmegaConf.is_config(conf):
        conf = OmegaConf.to_container(conf, resolve=True)
    conf = dict(conf) if isinstance(conf, dict) else conf
    if isinstance(conf, dict):
        if "_target_" in conf and conf["_target_"] in TARGET_REMAP:
            conf["_target_"] = TARGET_REMAP[conf["_target_"]]
        for k, v in list(conf.items()):
            conf[k] = remap_targets(v)
    elif isinstance(conf, list):
        conf = [remap_targets(v) for v in conf]
    return conf


def load_legacy_checkpoint(ckpt_path):
    """Load a legacy checkpoint and return model + config."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    state_dict = ckpt["state_dict"]

    network_conf = remap_targets(hp["network_conf"])

    # Rename 'network' -> 'featurizer' to match current BaseModule API
    if "network" in network_conf and "featurizer" not in network_conf:
        network_conf["featurizer"] = network_conf.pop("network")

    model = instantiate(network_conf, _convert_="all")

    # Remap state dict keys: model.network. -> featurizer., model.decoder. -> decoder.
    filtered = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("model."):
            key = key[len("model."):]
        if key.startswith("network."):
            key = "featurizer." + key[len("network."):]
        filtered[key] = v

    model.load_state_dict(filtered, strict=True)
    model.eval()
    return model, hp


def get_corpus_df(memmap_dir, split="test"):
    npz_path = os.path.join(memmap_dir, "metadata.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No metadata.npz found in {memmap_dir}")

    data = np.load(npz_path)
    def decode(arr):
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]

    records = []
    n = len(data["session_start_idx"])
    users = decode(data["users_user"])
    stages = decode(data.get("stages_stage", []))
    splits = decode(data.get("splits_split", ["train", "val", "test"]))

    for i in range(n):
        record = {
            "session_id": decode([data["session_session_id"][i]])[0],
            "user": users[int(data["session_user_id"][i])],
            "start_idx": int(data["session_start_idx"][i]),
            "end_idx": int(data["session_end_idx"][i]),
            "length": int(data["session_length"][i]),
        }
        if "session_stage_id" in data and len(stages) > 0:
            sid = int(data["session_stage_id"][i])
            record["stage"] = stages[sid] if sid < len(stages) else "unknown"
        if "session_split_id" in data:
            sid = int(data["session_split_id"][i])
            record["split"] = splits[sid] if sid < len(splits) else "unknown"
        else:
            record["split"] = split
        if "session_generalization" in data:
            record["generalization"] = decode([data["session_generalization"][i]])[0]
        else:
            if data.get("session_held_out_user", [False]*n)[i]:
                record["generalization"] = "cross_user"
            elif data.get("session_held_out_stage", [False]*n)[i]:
                record["generalization"] = "cross_stage"
            else:
                record["generalization"] = "seen_user"
        records.append(record)
    df = pd.DataFrame(records)
    if "split" in df.columns:
        df = df.query(f"split=='{split}'")
    return df


@torch.no_grad()
def evaluate(model, dataloader, device="cuda"):
    """Compute MAE incrementally to avoid OOM."""
    total_abs_err = 0.0
    total_valid = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        emg = batch["emg"].to(device)
        joint_angles = batch["joint_angles"].to(device)
        mask = batch["label_valid_mask"].to(device)

        batch_input = {"emg": emg, "joint_angles": joint_angles, "label_valid_mask": mask}
        result = model(batch_input)

        if isinstance(result, tuple):
            preds, targets, mask_t = result
        else:
            # VEMG2PoseWithInitialState returns just predictions
            preds = result
            start = model.left_context
            stop = None if model.right_context == 0 else -model.right_context
            targets = joint_angles[..., slice(start, stop)]
            mask_t = mask[..., slice(start, stop)]

        # Align time dims
        if preds.shape[-1] != targets.shape[-1]:
            preds = torch.nn.functional.interpolate(
                preds.unsqueeze(1) if preds.ndim == 2 else preds,
                size=targets.shape[-1], mode="linear"
            )

        # Ensure 3D: (batch, channels, time)
        if preds.ndim == 2:
            preds = preds.unsqueeze(1)
        if targets.ndim == 2:
            targets = targets.unsqueeze(1)

        n_channels = min(preds.shape[1], targets.shape[1])
        preds = preds[:, :n_channels]
        targets = targets[:, :n_channels]

        # Expand mask to match channel dims
        if mask_t.ndim == 1:
            mask_t = mask_t.unsqueeze(0).expand(preds.shape[0], -1)
        valid = mask_t.bool().unsqueeze(1).expand(-1, n_channels, -1)

        diff = torch.abs(preds - targets)
        total_abs_err += diff[valid].sum().item()
        total_valid += valid.sum().item()

    mae = total_abs_err / total_valid
    mae_deg = mae * 180.0 / np.pi
    return mae, mae_deg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to legacy checkpoint")
    parser.add_argument("--data-dir", default="/home/xiziheng/develop/emg2pose/data/emg_corpus/emg2pose_v3_memmap")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, hp = load_legacy_checkpoint(args.checkpoint)
    model.to(args.device)
    model.eval()

    print(f"Model: {type(model).__name__}")
    print(f"  left_context={model.left_context}, right_context={model.right_context}")
    print(f"  out_channels={model.out_channels}")

    memmap_dir = args.data_dir
    df = get_corpus_df(memmap_dir, split="test")

    effective_window = 10_000 + model.left_context + model.right_context
    transforms = Compose([ExtractToTensor(field="emg")])

    conditions = ["generalization"]
    groupby = df.groupby(conditions)

    results = []
    for vals, df_ in tqdm(groupby, desc="Conditions"):
        session_ids = df_["session_id"].tolist()

        dataset = Emg2PoseDataset(
            memmap_dir=memmap_dir,
            window_length=effective_window,
            stride=10_000,
            padding=(0, 0),
            jitter=False,
            skip_ik_failures=True,
            allowed_sessions=session_ids,
            transform=transforms,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

        mae, mae_deg = evaluate(model, dataloader, args.device)

        gen_name = vals if isinstance(vals, str) else vals[0]
        results.append({"generalization": gen_name, "mae_rad": mae, "mae_deg": mae_deg})
        print(f"  {gen_name}: MAE={mae:.4f} rad ({mae_deg:.2f} deg)")

    results_df = pd.DataFrame(results)
    ckpt_name = os.path.basename(args.checkpoint).replace(".ckpt", "")
    out_path = os.path.join(os.getcwd(), f"results_legacy_{ckpt_name}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
