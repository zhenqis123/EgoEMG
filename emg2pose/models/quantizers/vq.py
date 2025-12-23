from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """Vector quantization layer for VQ-VAE."""

    def __init__(
        self,
        num_codes: int,
        embed_dim: int,
        commitment_cost: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.embedding = nn.Embedding(num_codes, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
            self.register_buffer("ema_embedding", torch.zeros(num_codes, embed_dim))

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (N, D) float tensor.
        Returns:
            quantized: (N, D) quantized tensor (straight-through).
            indices: (N,) code indices.
            vq_loss: scalar tensor.
            codebook_loss: scalar tensor.
            commit_loss: scalar tensor.
        """
        if inputs.ndim != 2:
            raise ValueError(f"VectorQuantizer expects (N, D), got {inputs.shape}")

        inputs_sq = (inputs ** 2).sum(dim=1, keepdim=True)
        embed_sq = (self.embedding.weight ** 2).sum(dim=1)
        distances = inputs_sq + embed_sq - 2 * inputs @ self.embedding.weight.t()

        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices)

        if self.use_ema and self.training:
            # Memory-friendly EMA update without one-hot expansion.
            cluster_size = torch.bincount(indices, minlength=self.num_codes).type_as(
                inputs
            )
            embed_sum = torch.zeros(
                self.num_codes, self.embed_dim, device=inputs.device, dtype=inputs.dtype
            )
            embed_sum.index_add_(0, indices, inputs)
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1.0 - self.ema_decay
            )
            self.ema_embedding.mul_(self.ema_decay).add_(
                embed_sum, alpha=1.0 - self.ema_decay
            )
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.ema_eps) / (
                n + self.num_codes * self.ema_eps
            ) * n
            embed_normalized = self.ema_embedding / cluster_size.unsqueeze(1).clamp(
                min=self.ema_eps
            )
            self.embedding.weight.data.copy_(embed_normalized)

        codebook_loss = F.mse_loss(quantized, inputs.detach())
        commit_loss = F.mse_loss(inputs, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commit_loss

        quantized = inputs + (quantized - inputs).detach()
        return quantized, indices, vq_loss, codebook_loss, commit_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode code indices back to embeddings."""
        if indices.ndim != 1:
            raise ValueError(f"VectorQuantizer expects (N,), got {indices.shape}")
        return self.embedding(indices.long())

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight

    @staticmethod
    def compute_perplexity(indices: torch.Tensor, num_codes: int) -> torch.Tensor:
        flat = indices.reshape(-1)
        if flat.numel() == 0:
            return torch.tensor(0.0, device=flat.device)
        counts = torch.bincount(flat.long(), minlength=num_codes).float()
        probs = counts / counts.sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return torch.exp(entropy)


class GroupedVectorQuantizer(nn.Module):
    """Grouped VQ: split embedding dim into groups, each with its own codebook."""

    def __init__(
        self,
        num_groups: int,
        num_codes: int,
        group_dim: int,
        commitment_cost: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_codes = num_codes
        self.group_dim = group_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.chunk_size = chunk_size
        embed = torch.empty(num_groups, num_codes, group_dim)
        nn.init.uniform_(embed, -1.0 / num_codes, 1.0 / num_codes)
        self.embedding = nn.Parameter(embed)
        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_groups, num_codes))
            self.register_buffer("ema_embedding", torch.zeros(num_groups, num_codes, group_dim))

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2:
            raise ValueError(f"GroupedVectorQuantizer expects (N, D), got {inputs.shape}")
        if inputs.shape[1] != self.num_groups * self.group_dim:
            raise ValueError(
                f"Input dim {inputs.shape[1]} != num_groups*group_dim "
                f"{self.num_groups*self.group_dim}"
            )

        x = inputs.reshape(-1, self.num_groups, self.group_dim)
        embed_sq = (self.embedding ** 2).sum(dim=2)  # (G, C)
        chunk = self.chunk_size or x.shape[0]
        indices_list: list[torch.Tensor] = []
        quantized_list: list[torch.Tensor] = []
        if self.use_ema and self.training:
            cluster_size_acc = torch.zeros_like(self.ema_cluster_size)
            embed_sum_acc = torch.zeros_like(self.ema_embedding)
        for start in range(0, x.shape[0], chunk):
            end = min(start + chunk, x.shape[0])
            x_chunk = x[start:end]
            inputs_sq = (x_chunk ** 2).sum(dim=2, keepdim=True)  # (Nc, G, 1)
            distances = inputs_sq + embed_sq.unsqueeze(0) - 2 * torch.einsum(
                "ngd,gcd->ngc", x_chunk, self.embedding
            )
            indices = torch.argmin(distances, dim=2).long()  # (Nc, G)
            quantized = torch.stack(
                [
                    self.embedding[g].index_select(0, indices[:, g])
                    for g in range(self.num_groups)
                ],
                dim=1,
            )  # (Nc, G, Dg)
            indices_list.append(indices)
            quantized_list.append(quantized)
            if self.use_ema and self.training:
                ones = torch.ones(
                    indices.shape[0],
                    device=indices.device,
                    dtype=cluster_size_acc.dtype,
                )
                for g in range(self.num_groups):
                    idx = indices[:, g]
                    cluster_size_acc[g].index_add_(0, idx, ones)
                    embed_sum_acc[g].index_add_(0, idx, x_chunk[:, g, :])

        indices = torch.cat(indices_list, dim=0)
        quantized = torch.cat(quantized_list, dim=0)
        quantized_flat = quantized.reshape(inputs.shape)
        
        if self.use_ema and self.training:
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size_acc, alpha=1.0 - self.ema_decay
            )
            self.ema_embedding.mul_(self.ema_decay).add_(
                embed_sum_acc, alpha=1.0 - self.ema_decay
            )
            n = self.ema_cluster_size.sum(dim=1, keepdim=True)  # (G,1)
            cluster_norm = (self.ema_cluster_size + self.ema_eps) / (
                n + self.num_codes * self.ema_eps
            ) * n
            embed_norm = self.ema_embedding / cluster_norm.unsqueeze(-1).clamp(
                min=self.ema_eps
            )
            self.embedding.data.copy_(embed_norm)

        codebook_loss = F.mse_loss(quantized_flat, inputs.detach())
        commit_loss = F.mse_loss(inputs, quantized_flat.detach())
        vq_loss = codebook_loss + self.commitment_cost * commit_loss

        quantized_flat = inputs + (quantized_flat - inputs).detach()
        return quantized_flat, indices, vq_loss, codebook_loss, commit_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode grouped indices back to embeddings."""
        if indices.ndim != 2:
            raise ValueError(f"GroupedVectorQuantizer expects (N, G), got {indices.shape}")
        if indices.shape[1] != self.num_groups:
            raise ValueError(
                f"Indices groups {indices.shape[1]} != num_groups {self.num_groups}"
            )
        indices = indices.long()
        quantized = torch.stack(
            [
                self.embedding[g].index_select(0, indices[:, g])
                for g in range(self.num_groups)
            ],
            dim=1,
        )
        return quantized.reshape(indices.shape[0], -1)

    def get_codebook(self) -> torch.Tensor:
        return self.embedding

    @staticmethod
    def compute_perplexity(indices: torch.Tensor, num_codes: int) -> torch.Tensor:
        flat = indices.reshape(-1)
        if flat.numel() == 0:
            return torch.tensor(0.0, device=flat.device)
        counts = torch.bincount(flat.long(), minlength=num_codes).float()
        probs = counts / counts.sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return torch.exp(entropy)


class ResidualVectorQuantizer(nn.Module):
    """Residual VQ: multi-level codebooks sequentially quantize residuals."""

    def __init__(
        self,
        num_levels: int,
        num_codes: int,
        embed_dim: int,
        commitment_cost: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        embed = torch.empty(num_levels, num_codes, embed_dim)
        nn.init.uniform_(embed, -1.0 / num_codes, 1.0 / num_codes)
        self.embedding = nn.Parameter(embed)
        if self.use_ema:
            self.register_buffer(
                "ema_cluster_size", torch.zeros(num_levels, num_codes)
            )
            self.register_buffer(
                "ema_embedding", torch.zeros(num_levels, num_codes, embed_dim)
            )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2:
            raise ValueError(f"ResidualVectorQuantizer expects (N, D), got {inputs.shape}")
        if inputs.shape[1] != self.embed_dim:
            raise ValueError(
                f"Input dim {inputs.shape[1]} != embed_dim {self.embed_dim}"
            )

        residual = inputs
        quantized_levels = []
        indices_per_level = []
        codebook_loss = 0.0
        commit_loss = 0.0

        for lvl in range(self.num_levels):
            embed = self.embedding[lvl]  # (C, D)
            res_sq = (residual ** 2).sum(dim=1, keepdim=True)
            embed_sq = (embed ** 2).sum(dim=1)
            distances = res_sq + embed_sq - 2 * residual @ embed.t()
            indices = torch.argmin(distances, dim=1)
            quantized = embed.index_select(0, indices)

            if self.use_ema and self.training:
                one_hot = F.one_hot(indices, num_classes=self.num_codes).type_as(inputs)
                cluster_size = one_hot.sum(dim=0)
                embed_sum = one_hot.t() @ residual
                self.ema_cluster_size[lvl].mul_(self.ema_decay).add_(
                    cluster_size, alpha=1.0 - self.ema_decay
                )
                self.ema_embedding[lvl].mul_(self.ema_decay).add_(
                    embed_sum, alpha=1.0 - self.ema_decay
                )
                n = self.ema_cluster_size[lvl].sum()
                cluster_norm = (self.ema_cluster_size[lvl] + self.ema_eps) / (
                    n + self.num_codes * self.ema_eps
                ) * n
                embed_norm = self.ema_embedding[lvl] / cluster_norm.unsqueeze(-1).clamp(
                    min=self.ema_eps
                )
                self.embedding.data[lvl].copy_(embed_norm)

            codebook_loss = codebook_loss + F.mse_loss(quantized, residual.detach())
            commit_loss = commit_loss + F.mse_loss(residual, quantized.detach())

            quantized_levels.append(quantized)
            indices_per_level.append(indices)
            residual = residual - quantized.detach()

        quantized_sum = torch.stack(quantized_levels, dim=0).sum(dim=0)
        quantized_sum = inputs + (quantized_sum - inputs).detach()

        vq_loss = codebook_loss + self.commitment_cost * commit_loss
        indices_tensor = torch.stack(indices_per_level, dim=0)  # (L, N)
        return quantized_sum, indices_tensor, vq_loss, codebook_loss, commit_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode residual code indices back to quantized embeddings."""
        if indices.ndim == 1:
            if self.num_levels != 1:
                raise ValueError(
                    f"Expected (N, L) with L={self.num_levels}, got {indices.shape}"
                )
            indices = indices.unsqueeze(1)
        if indices.ndim != 2:
            raise ValueError(f"ResidualVectorQuantizer expects (N, L), got {indices.shape}")
        if indices.shape[0] == self.num_levels and indices.shape[1] != self.num_levels:
            indices = indices.transpose(0, 1)
        if indices.shape[1] != self.num_levels:
            raise ValueError(
                f"Indices levels {indices.shape[1]} != num_levels {self.num_levels}"
            )
        indices = indices.long()
        quantized_levels = []
        for lvl in range(self.num_levels):
            embed = self.embedding[lvl]
            quantized_levels.append(embed.index_select(0, indices[:, lvl]))
        return torch.stack(quantized_levels, dim=0).sum(dim=0)

    def get_codebook(self) -> torch.Tensor:
        return self.embedding

    @staticmethod
    def compute_perplexity(indices: torch.Tensor, num_codes: int) -> torch.Tensor:
        flat = indices.reshape(-1)
        if flat.numel() == 0:
            return torch.tensor(0.0, device=flat.device)
        counts = torch.bincount(flat.long(), minlength=num_codes).float()
        probs = counts / counts.sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return torch.exp(entropy)
