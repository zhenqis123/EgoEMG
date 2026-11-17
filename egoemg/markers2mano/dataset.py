"""Processed MANO mesh HDF5/memmap reader; source training datasets are not shipped."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class HandDatasetSimple(Dataset):
    """
    Eager-loading Dataset for processed HanCo-style data.
    一次性把所有 split.h5 的数据读入内存（vertices/poses(or thetas)/betas），
    并在 __getitem__ 中直接返回 torch.Tensor。
    """

    def __init__(
        self,
        data_root: str,
        data_name_list: List[str],
        split: str = 'train',
        transform=None,
        dtype=np.float32,
        keep_indices: bool = False,
        require_poses: bool = True,
        require_betas: bool = True,
        use_mmap: bool = True,
    ):
        """
        Args:
            data_root: 根目录，包含若干子目录，每个子目录下有 {split}.h5；
                       若 use_mmap=True，则优先从 {data_root}/merged/ 加载预合并 .npy
            data_name_list: 子目录名列表（mmap 模式下仅用于记录）
            split: 'train' 或 'val' 等
            transform: 可选变换函数，输入/输出均为 dict
            dtype: numpy 读入 dtype（默认 float32）
            keep_indices: 是否保留原文件内的样本索引映射
            require_poses: 若为 True，且文件中无 poses/thetas 则报错
            require_betas: 若为 True，且文件中无 betas 则报错
            use_mmap: 若为 True，优先从 merged/ 目录 mmap 加载 .npy
        """
        self.markers_indices = torch.tensor(
            [191, 88, 253, 708, 729, 144, 87, 295, 319, 220,
             365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673],
            dtype=torch.long
        )

        self.data_name_list = list(data_name_list)
        self.data_path_list = [Path(data_root) / name for name in data_name_list]
        self.split = split
        self.transform = transform
        self.keep_indices = keep_indices
        self.require_poses = require_poses
        self.require_betas = require_betas
        self.use_mmap = use_mmap

        if use_mmap:
            merged_dir = Path(data_root) / "merged"
            vert_path = merged_dir / f"{split}_vertices.npy"
            if vert_path.exists():
                self.vertices = np.load(str(vert_path), mmap_mode="r")
                pose_path = merged_dir / f"{split}_poses.npy"
                beta_path = merged_dir / f"{split}_betas.npy"
                self.poses = np.load(str(pose_path), mmap_mode="r") if pose_path.exists() else None
                self.betas = np.load(str(beta_path), mmap_mode="r") if beta_path.exists() else None
                count = len(self.vertices)
                if self.vertices.shape[1:] != (778, 3):
                    raise ValueError("Expected vertices with shape (N,778,3), in metres")
                for name, array, required in [
                    ("poses", self.poses, self.require_poses),
                    ("betas", self.betas, self.require_betas),
                ]:
                    if required and array is None:
                        raise FileNotFoundError(f"Missing {split}_{name}.npy")
                    if array is not None and len(array) != count:
                        raise ValueError(f"Length mismatch for {name}")
                self.sample_source = [("merged", i) for i in range(count)] if keep_indices else []
                print(f"[mmap] Loaded {split}: {self.vertices.shape[0]} samples "
                      f"(verts {self.vertices.shape}, poses {'ok' if self.poses is not None else 'None'}, "
                      f"betas {'ok' if self.betas is not None else 'None'})")
                return
            else:
                print(f"[mmap] {vert_path} not found, falling back to h5 loading")

        (self.vertices,
         self.poses,
         self.betas,
         self.sample_source) = self._load_all(dtype=dtype)

    # ----------------------------- IO helpers -----------------------------

    @staticmethod
    def _get_first_existing_key(h5f: h5py.File, keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in h5f:
                return k
        return None

    def _load_one_file(self, data_file: str, data_name: str, dtype=np.float32
                        ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        读取单个 {split}.h5，并返回 (verts, poses_or_thetas, betas, valid_idx)
        - 若某个字段不存在且不要求（require_* 为 False），返回 None
        - 做联合有效性过滤（NaN/Inf）
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Processed data file not found: {data_file}")

        with h5py.File(data_file, 'r') as f:
            if 'vertices' not in f:
                raise KeyError(f"'vertices' dataset not found in {data_file}")

            verts = f['vertices'][:]  # (N, 778, 3)
            if verts.ndim != 3 or verts.shape[1:] != (778, 3):
                raise ValueError(f"Invalid vertices shape in {data_file}: {verts.shape}, expected (N, 778, 3)")

            # poses or thetas
            poses_key = self._get_first_existing_key(f, ['poses', 'thetas'])
            poses = None
            if poses_key is not None:
                poses = f[poses_key][:]  # (N, D) 期望 D=48
                if poses.ndim != 2:
                    # raise ValueError(f"Invalid {poses_key} shape in {data_file}: {poses.shape}, expected (N, D)")
                    poses = poses.reshape(poses.shape[0], -1)  # 自动调整维度
            elif self.require_poses:
                raise KeyError(f"'poses' or 'thetas' not found in {data_file}")

            # betas
            betas = None
            if 'betas' in f:
                betas = f['betas'][:]  # (N, 10)
                if betas.ndim != 2:
                    # raise ValueError(f"Invalid betas shape in {data_file}: {betas.shape}, expected (N, 10)")
                    betas = betas.reshape(betas.shape[0], -1)  # 自动调整维度
            elif self.require_betas:
                raise KeyError(f"'betas' dataset not found in {data_file}")

            # --------- 联合有效性过滤（去除 NaN/Inf）---------
            valid_mask = np.isfinite(verts).all(axis=(1, 2))
            if poses is not None:
                valid_mask &= np.isfinite(poses).all(axis=1)
            if betas is not None:
                valid_mask &= np.isfinite(betas).all(axis=1)

            num_total = verts.shape[0]
            num_bad = int((~valid_mask).sum())
            if num_bad > 0:
                print(f"[{data_name}] filtered {num_bad}/{num_total} invalid samples (nan/inf)")

            # 应用过滤
            verts = verts[valid_mask].astype(dtype, copy=False)
            if poses is not None:
                poses = poses[valid_mask].astype(dtype, copy=False)
            if betas is not None:
                betas = betas[valid_mask].astype(dtype, copy=False)

            valid_idx = np.nonzero(valid_mask)[0]  # 源文件中的局部索引

        return verts, poses, betas, valid_idx

    def _load_all(self, dtype=np.float32):
        verts_all = []
        poses_all = []
        betas_all = []
        sample_source = []  # [(data_name, local_idx), ...]


        for data_path, data_name in tqdm(
            zip(self.data_path_list, self.data_name_list),
            total=len(self.data_path_list),
            desc="Loading data",
        ):
            data_file = os.path.join(str(data_path), f"{self.split}.h5")
            verts, poses, betas, valid_idx = self._load_one_file(data_file, data_name, dtype=dtype)

            verts_all.append(verts)
            if poses is not None:
                poses_all.append(poses)
            else:
                poses_all.append(None)

            if betas is not None:
                betas_all.append(betas)
            else:
                betas_all.append(None)

            if self.keep_indices:
                sample_source.extend([(data_name, int(i)) for i in valid_idx])
            else:
                sample_source.extend([(data_name, -1)] * verts.shape[0])

        if len(verts_all) == 0:
            raise RuntimeError("No valid vertices loaded. Please check your data paths.")

        # 预分配 + 逐块填入（避免 np.concatenate 的额外内存分配和拷贝）
        def _prealloc_concat(parts: List[Optional[np.ndarray]], label: str = "") -> Optional[np.ndarray]:
            present = [p for p in parts if p is not None]
            if not present:
                return None
            total_n = sum(p.shape[0] for p in present)
            shape = (total_n,) + present[0].shape[1:]
            out = np.empty(shape, dtype=present[0].dtype)
            offset = 0
            for i, p in tqdm(enumerate(present), total=len(present),
                              desc=f"Merging {label}", leave=False):
                n = p.shape[0]
                out[offset:offset + n] = p
                present[i] = None  # 立即释放
                offset += n
            return out

        vertices_all = _prealloc_concat(verts_all, "vertices")
        del verts_all
        poses_cat = (_prealloc_concat(poses_all, "poses")
                     if all(p is not None for p in poses_all) else None)
        del poses_all
        betas_cat = (_prealloc_concat(betas_all, "betas")
                     if all(p is not None for p in betas_all) else None)
        del betas_all

        # 转 torch tensor（zero-copy，数据已是 float32）
        vertices_all = torch.from_numpy(vertices_all)
        poses_tensor = torch.from_numpy(poses_cat) if poses_cat is not None else None
        betas_tensor = torch.from_numpy(betas_cat) if betas_cat is not None else None

        # 断言长度一致（若存在）
        N = vertices_all.shape[0]
        if poses_tensor is not None and poses_tensor.shape[0] != N:
            raise RuntimeError(f"Length mismatch: verts(N={N}) vs poses(N={poses_tensor.shape[0]})")
        if betas_tensor is not None and betas_tensor.shape[0] != N:
            raise RuntimeError(f"Length mismatch: verts(N={N}) vs betas(N={betas_tensor.shape[0]})")

        return vertices_all, poses_tensor, betas_tensor, sample_source

    # ----------------------------- dataset API -----------------------------

    def __len__(self):
        return self.vertices.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # mmap "r" 模式返回只读数组，需要 copy 才能转 torch tensor
        _get = lambda arr, i: torch.from_numpy(arr[i].copy()) if isinstance(arr, np.ndarray) else arr[i]

        sample = {
            'verts': _get(self.vertices, idx),  # (778, 3) float32
        }
        if self.poses is not None:
            sample['poses'] = _get(self.poses, idx)    # (D,)
        if self.betas is not None:
            sample['betas'] = _get(self.betas, idx)    # (10,)

        if self.keep_indices:
            data_name, local_idx = self.sample_source[idx]
            sample['source'] = {'data_name': data_name, 'local_idx': local_idx}

        if self.transform is not None:
            sample = self.transform(sample)
        if 'poses' not in sample or torch.count_nonzero(sample['poses']) == 0:
            sample['mano_para_available'] = torch.zeros(1)
        else:
            sample['mano_para_available'] = torch.ones(1)
        return sample
