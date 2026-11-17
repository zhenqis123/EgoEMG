"""Marker corruption and mesh-neighborhood blending migrated from the source trainer."""

import torch


class MarkerAugmentation:
    SKELETON_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (0, 9), (9, 10), (10, 11), (11, 12),     # middle
        (0, 13), (13, 14), (14, 15), (15, 16),   # ring
        (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    ]

    SKELETON_ADJACENCY = {
        0: [1, 5, 9, 13, 17],
        1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3],
        5: [0, 6], 6: [5, 7], 7: [6, 8], 8: [7],
        9: [0, 10], 10: [9, 11], 11: [10, 12], 12: [11],
        13: [0, 14], 14: [13, 15], 15: [14, 16], 16: [15],
        17: [0, 18], 18: [17, 19], 19: [18, 20], 20: [19],
    }

    def _augment_bone_length(
        self, markers: torch.Tensor, verts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perturb bone lengths along skeleton edges to simulate soft-tissue artifact."""
        bone_scale_range = float(self.hparams.bone_scale_range)
        if bone_scale_range <= 0:
            return markers, verts

        out = markers.clone()
        B = out.shape[0]
        device = out.device

        # Precompute descendants for each child joint
        for parent, child in self.SKELETON_EDGES:
            # Decide which samples perturb this bone: (B,)
            do_perturb = torch.rand(B, device=device) < 0.3
            if not do_perturb.any():
                continue

            scale = 1.0 + (torch.rand(B, 1, device=device) * 2 - 1) * bone_scale_range
            bone_vec = out[:, child] - out[:, parent]  # (B, 3)
            offset = bone_vec * (scale - 1.0)  # (B, 3)
            offset = offset * do_perturb.unsqueeze(1).float()  # zero out non-perturbed

            descendants = self._descendants_cache.get(child, [child])
            for d in descendants:
                out[:, d] = out[:, d] + offset

        return out, verts

    def _build_descendants(self, joint_idx: int) -> list[int]:
        """Build list of joint_idx and all its descendants in the skeleton tree."""
        descendants = [joint_idx]
        stack = [joint_idx]
        while stack:
            curr = stack.pop()
            for child_parent, child in self.SKELETON_EDGES:
                if child_parent == curr and child not in descendants:
                    descendants.append(child)
                    stack.append(child)
        return descendants

    def _augment_swap(self, markers: torch.Tensor) -> torch.Tensor:
        """Swap positions of markers that are spatially close (all-pairs via cdist)."""
        swap_distance_threshold = float(self.hparams.swap_distance_threshold)
        swap_prob = float(self.hparams.swap_prob)
        max_swaps = int(self.hparams.max_swaps_per_frame)
        if swap_distance_threshold <= 0 or swap_prob <= 0:
            return markers

        out = markers.clone()
        B, N, _ = out.shape
        device = out.device

        # All-pairs distance: (B, N, N), fully vectorized
        dist_matrix = torch.cdist(out, out, p=2)  # (B, 21, 21)
        ii, jj = torch.triu_indices(N, N, offset=1, device=device)

        for _ in range(max_swaps):
            # For each sample, find the closest pair among remaining close pairs
            dists = dist_matrix[:, ii, jj]  # (B, num_pairs)

            # Mask: only consider pairs within threshold
            close_mask = dists < swap_distance_threshold  # (B, num_pairs)
            if not close_mask.any():
                break

            # Decide swap: per sample
            do_swap = torch.rand(B, device=device) < swap_prob  # (B,)

            # For each sample, pick one random close pair to swap
            # Set non-close distances to inf so argmin won't pick them
            masked_dists = dists.clone()
            masked_dists[~close_mask] = float('inf')
            # Random tie-breaking: add small noise
            masked_dists = masked_dists + torch.rand_like(masked_dists) * 1e-6
            pair_idx = masked_dists.argmin(dim=1)  # (B,)

            # Perform swap for qualifying samples
            for b in range(B):
                if not do_swap[b] or not close_mask[b].any():
                    continue
                p = pair_idx[b].item()
                i, j = ii[p].item(), jj[p].item()
                tmp = out[b, i].clone()
                out[b, i] = out[b, j]
                out[b, j] = tmp
                # Update distance matrix for this sample (save old before overwrite)
                old_row_i = dist_matrix[b, i, :].clone()
                old_col_i = dist_matrix[b, :, i].clone()
                dist_matrix[b, i, :] = dist_matrix[b, j, :]
                dist_matrix[b, :, i] = dist_matrix[b, :, j]
                dist_matrix[b, j, :] = old_row_i
                dist_matrix[b, :, j] = old_col_i

        return out

    def _augment_drift(self, markers: torch.Tensor) -> torch.Tensor:
        """Apply systematic spatial offset to a random subset of markers (fully vectorized)."""
        magnitude = float(self.hparams.marker_drift_magnitude)
        num_drifted = int(self.hparams.num_drifted_markers)
        if magnitude <= 0 or num_drifted <= 0:
            return markers

        B, N, C = markers.shape
        device = markers.device
        out = markers.clone()

        # Select random markers: use argsort of random matrix (no Python loop)
        rand_mat = torch.rand(B, N, device=device)
        indices = rand_mat.argsort(dim=1)[:, :num_drifted]  # (B, num_drifted)

        # Random magnitudes and directions: (B, num_drifted, 3)
        drift_mag = torch.rand(B, num_drifted, 1, device=device) * magnitude
        directions = torch.randn(B, num_drifted, C, device=device)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        offsets = directions * drift_mag

        # Apply via advanced indexing
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, num_drifted)
        out[batch_idx, indices] = out[batch_idx, indices] + offsets

        return out

    def _augment_spike(self, markers: torch.Tensor) -> torch.Tensor:
        """Displace a random marker to an anomalously far position (vectorized)."""
        spike_prob = float(self.hparams.spike_prob)
        if spike_prob <= 0:
            return markers

        B, N, C = markers.shape
        device = markers.device
        out = markers.clone()

        # Decide which samples get spiked
        do_spike = torch.rand(B, device=device) < spike_prob
        if not do_spike.any():
            return out

        spiked_samples = do_spike.nonzero().squeeze(-1)
        spike_B = spiked_samples.shape[0]

        # Hand scale
        wrist = out[:, 0:1, :]  # (B, 1, 3)
        hand_scale = (out - wrist).norm(dim=-1).mean(dim=-1)  # (B,)

        # Random marker index (skip wrist=0)
        spike_idx = torch.randint(1, N, (spike_B,), device=device)
        # Random scale
        scale_min = float(self.hparams.spike_scale_min)
        scale_max = float(self.hparams.spike_scale_max)
        spike_scale = torch.rand(spike_B, device=device) * (scale_max - scale_min) + scale_min
        # Random direction
        direction = torch.randn(spike_B, C, device=device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # New positions
        new_pos = out[spiked_samples, 0] + direction * spike_scale.unsqueeze(1) * hand_scale[spiked_samples].unsqueeze(1)
        out[spiked_samples, spike_idx] = new_pos

        return out

    def _augment_dropout(self, markers: torch.Tensor) -> torch.Tensor:
        """Drop random markers and replace with neighbor-interpolated positions (fully vectorized)."""
        dropout_prob = float(self.hparams.marker_dropout_prob)
        max_dropout = int(self.hparams.max_dropout)
        if dropout_prob <= 0:
            return markers

        B, N, C = markers.shape
        device = markers.device
        out = markers.clone()

        # Decide which markers to drop: (B, N)
        drop_mask = torch.rand(B, N, device=device) < dropout_prob

        # Cap at max_dropout per sample: add random noise and keep top-k
        num_dropped = drop_mask.sum(dim=1)
        excess = num_dropped > max_dropout
        if excess.any():
            # For excess samples, re-sample using top-k of random values
            rand_vals = torch.rand(B, N, device=device)
            rand_vals[~drop_mask] = -1.0  # non-dropped get low value
            # Use kthvalue to find threshold per sample
            k = max_dropout
            thresholds = rand_vals.kthvalue(N - k, dim=1).values.unsqueeze(1)  # (B, 1)
            drop_mask = drop_mask & (rand_vals >= thresholds)

        if not drop_mask.any():
            return out

        # Build neighbor weight matrix: (N, N) binary adjacency
        # We precompute this once as a class attribute
        if not hasattr(self, '_neighbor_weight'):
            adj = torch.zeros(N, N)
            for idx, nbrs in self.SKELETON_ADJACENCY.items():
                for n in nbrs:
                    adj[idx, n] = 1.0
            # Self-loops excluded: a dropped marker cannot interpolate from itself
            self._neighbor_weight = adj  # (N, N)

        adj = self._neighbor_weight.to(device)  # (N, N)

        # Valid neighbor mask: neighbor is NOT dropped → (B, N, N)
        valid_nbr = ~drop_mask.unsqueeze(1)  # (B, 1, N) broadcast to (B, N, N)
        valid_adj = adj.unsqueeze(0) * valid_nbr.float()  # (B, N, N)

        # Weighted average of valid neighbors
        nbr_sum = torch.bmm(valid_adj, out)  # (B, N, 3)
        nbr_count = valid_adj.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, N, 1)
        interp = nbr_sum / nbr_count  # (B, N, 3)

        # Only replace dropped markers with valid neighbors
        has_valid = (valid_adj.sum(dim=-1) > 0)  # (B, N)
        replace_mask = drop_mask & has_valid  # (B, N)
        out[replace_mask] = interp[replace_mask]

        return out

    def _prepare_input(
        self, verts: torch.Tensor, markers: torch.Tensor, augment: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Data augmentation pipeline:
        1) Bone length perturbation (structural)
        2) Marker swap (identity confusion, all-pairs distance)
        3) Marker drift (systematic bias)
        4) Marker spike (anomalous outlier position)
        5) Neighborhood blending + Gaussian noise (existing)
        6) Marker dropout (occlusion)
        7) Root-centering + random scaling (existing)

        当 ``augment=False`` 时（验证/推理），全部 1-7 步增强都会跳过，
        只做必要的 dtype/device 对齐和 root-centering，保证 val loss 稳定且
        反映真实无扰动误差。
        """
        if markers is None:
            markers = verts[:, self.markers_indices, :]

        verts = verts.float()
        markers = markers.to(dtype=verts.dtype, device=verts.device)
        device = verts.device
        batch_size = verts.shape[0]
        num_markers = markers.shape[1]

        if augment:
            # 1. Bone length perturbation
            markers, verts = self._augment_bone_length(markers, verts)
            # 2. Marker swap
            markers = self._augment_swap(markers)
            # 3. Marker drift
            markers = self._augment_drift(markers)
            # 4. Marker spike (anomalous outlier)
            markers = self._augment_spike(markers)

        # 邻域混合：用相邻 MANO 顶点对 marker 位置做加权平滑。
        # 注意 rand_weights 是随机的，属于增强；val/augment=False 时退化为纯 marker。
        marker_self_weight = getattr(self.hparams, "marker_self_weight", 0.5)
        marker_self_weight = float(max(0.0, min(1.0, marker_self_weight)))
        if not augment:
            marker_self_weight = 1.0
        neighbor_weight = 1.0 - marker_self_weight

        blended_markers = torch.empty(
            batch_size, num_markers, 3, device=device, dtype=verts.dtype
        )

        for local_idx, mano_idx in enumerate(self.markers_indices.tolist()):
            neighbors = self.marker_neighbor_indices.get(int(mano_idx))
            if (
                neighbors is None
                or neighbors.numel() == 0
                or neighbor_weight <= 1e-6
            ):
                blended_markers[:, local_idx, :] = markers[:, local_idx, :]
                continue

            neighbor_ids = neighbors.to(device)
            neighbor_positions = torch.index_select(verts, 1, neighbor_ids)
            # 去掉 marker 自身，避免双重计数
            valid_mask = neighbor_ids != mano_idx
            neighbor_positions = neighbor_positions[:, valid_mask, :]

            if neighbor_positions.shape[1] == 0:
                blended_markers[:, local_idx, :] = markers[:, local_idx, :]
                continue

            rand_weights = torch.rand(
                batch_size,
                neighbor_positions.shape[1],
                1,
                device=device,
                dtype=verts.dtype,
            )
            rand_weights = rand_weights / rand_weights.sum(dim=1, keepdim=True).clamp_(
                min=1e-6
            )
            neighbor_mix = torch.sum(neighbor_positions * rand_weights, dim=1)
            blended_markers[:, local_idx, :] = (
                marker_self_weight * markers[:, local_idx, :]
                + neighbor_weight * neighbor_mix
            )

        # 5. Neighborhood blending 后加高斯噪声 + dropout mask
        if augment:
            noise_std = float(self.hparams.marker_noise_std)
            if noise_std > 0:
                noise = torch.randn_like(blended_markers) * noise_std
                drop_prob = float(self.hparams.noise_dropout_prob)
                if drop_prob > 0:
                    mask = (
                        torch.rand(
                            batch_size,
                            num_markers,
                            1,
                            device=device,
                            dtype=verts.dtype,
                        )
                        > drop_prob
                    ).float()
                    noise = noise * mask
                blended_markers = blended_markers + noise

            # 6. Marker dropout (after noise, so interpolation uses noisy positions)
            blended_markers = self._augment_dropout(blended_markers)

        # 7. Root-centering（确定性，train/val 都做）
        root_vertex_idx = int(self.markers_indices[0].item())
        root = verts[:, root_vertex_idx, :].unsqueeze(1).clone()
        verts_local = verts - root
        markers_local = blended_markers - root

        # 8. 随机 scale（增强）：val 时跳过
        if augment:
            scale_range = getattr(self.hparams, "marker_scale_range", 0.5)
            if scale_range > 0:
                # scale ∈ [1 - r, 1 + r]，每个样本独立
                scales = 1.0 + (torch.rand(batch_size, 1, 1, device=device) * 2 - 1) * scale_range
                verts_local = verts_local * scales
                markers_local = markers_local * scales

        return markers_local, verts_local
