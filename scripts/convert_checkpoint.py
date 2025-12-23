#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import torch


def remap_key(k: str) -> str:
    # 仅按你报错里看到的变化做重命名
    if k.startswith("model.network."):
        return "model.featurizer." + k[len("model.network."):]
    if k == "model.network":
        return "model.featurizer"
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True, type=str)
    ap.add_argument("--out_weights", required=True, type=str, help="输出 .pth/.pt 文件")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strip_prefix", default="", help="可选：移除统一前缀，比如 'model.'")
    args = ap.parse_args()

    ckpt = torch.load(args.in_ckpt, map_location=args.device)
    sd = ckpt["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        k2 = remap_key(k)
        if args.strip_prefix and k2.startswith(args.strip_prefix):
            k2 = k2[len(args.strip_prefix):]
        new_sd[k2] = v

    out_path = Path(args.out_weights)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, out_path)
    print(f"[OK] saved weights-only state_dict to: {out_path}")
    print(f"[INFO] num params: {len(new_sd)}")


if __name__ == "__main__":
    main()
