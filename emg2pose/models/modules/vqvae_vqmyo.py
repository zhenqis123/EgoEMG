"""
Self-contained Joint Angle VQ-VAE (2D Conv) for tokenizing hand joint-angle trajectories.

Input:
    x_3d: torch.Tensor of shape (B, 5, 4, 2000)

Output:
    recon_20T: torch.Tensor of shape (B, 20, 2000)
    indices:   torch.Tensor of shape (B, 5, 200)   (token id per (joint-pos, time))
    vq_loss: torch.Tensor scalar
    codebook_loss: torch.Tensor scalar
    commit_loss: torch.Tensor scalar

Design matches the paper description:
- Reshape inputs to 5×4×T (already provided).
- Encoder: stacked 2D convolutions, strides restricted to temporal axis only.
- Latent: (D=10, H=5, W=200) from input T=2000 (downsample factor 10).
- Vector quantization per spatial position (H×W).
- Decoder mirrors encoder with upsampling only along temporal axis.
- Reconstruct back to (5×4×T), reshape to (20×T), then apply a 1×1 "mixing" Conv1d across 20 channels.

No external imports besides torch.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# -------------------------
# Utility
# -------------------------
def _crop_or_pad_time(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Force x's last dimension to be exactly target_T by cropping or right-padding zeros."""
    w = x.shape[-1]
    if w == target_T:
        return x
    if w > target_T:
        return x[..., :target_T]
    return F.pad(x, (0, target_T - w))


# -------------------------
# Vector Quantizer (VQ-VAE)
# -------------------------
class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE vector quantizer.
    Expects inputs z of shape (B, N, D). Returns quantized z_q same shape and indices (B, N).

    Loss decomposition:
      codebook_loss = ||sg[z_e] - e||^2
      commit_loss   = beta * ||z_e - sg[e]||^2
      vq_loss       = codebook_loss + commit_loss
    """

    def __init__(self, num_codes: int, embed_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.embed_dim = int(embed_dim)
        self.commitment_cost = float(commitment_cost)

        self.codebook = nn.Embedding(self.num_codes, self.embed_dim)
        # common init
        nn.init.uniform_(self.codebook.weight, -1.0 / self.num_codes, 1.0 / self.num_codes)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, N, D)
        """
        if z.ndim != 3:
            raise ValueError(f"VectorQuantizer expects (B, N, D), got {tuple(z.shape)}")

        b, n, d = z.shape
        if d != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: z has {d}, codebook has {self.embed_dim}")

        # Flatten for distance computation: (B*N, D)
        z_flat = z.reshape(b * n, d)

        # Compute squared L2 distance to codebook entries:
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        # z_norm: (B*N, 1), e_norm: (K,), dot: (B*N, K)
        z_norm = (z_flat ** 2).sum(dim=1, keepdim=True)  # (B*N, 1)
        e = self.codebook.weight                          # (K, D)
        e_norm = (e ** 2).sum(dim=1)                      # (K,)
        dot = z_flat @ e.t()                              # (B*N, K)
        dist = z_norm + e_norm.unsqueeze(0) - 2.0 * dot   # (B*N, K)

        # Nearest codebook index
        indices = torch.argmin(dist, dim=1)               # (B*N,)
        indices_2d = indices.view(b, n)                   # (B, N)

        # Quantize
        z_q = self.codebook(indices).view(b, n, d)        # (B, N, D)

        # Losses
        # codebook loss: move codebook toward encoder outputs (stopgrad on encoder)
        codebook_loss = F.mse_loss(z_q, z.detach())
        # commitment loss: encourage encoder outputs to commit to codebook (stopgrad on codebook)
        commit_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + commit_loss

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        return z_q_st, indices_2d, vq_loss, codebook_loss, commit_loss


class VQ2DWrapper(nn.Module):
    """
    Quantize a 2D feature map z_e of shape (B, D, H, W) by flattening (H*W) positions.
    Returns:
      z_q:     (B, D, H, W)
      indices: (B, H, W)
      losses:  scalars
    """

    def __init__(self, vq: VectorQuantizer) -> None:
        super().__init__()
        self.vq = vq
        self.num_codes = 128

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_e.ndim != 4:
            raise ValueError(f"VQ2DWrapper expects (B, D, H, W), got {tuple(z_e.shape)}")
        b, d, h, w = z_e.shape

        # (B, D, H, W) -> (B, H*W, D)
        z = z_e.permute(0, 2, 3, 1).contiguous().view(b, h * w, d)

        z_q, indices, vq_loss, codebook_loss, commit_loss = self.vq(z)

        # (B, H*W, D) -> (B, D, H, W)
        z_q = z_q.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        indices = indices.view(b, h, w)

        return z_q, indices, vq_loss, codebook_loss, commit_loss


# -------------------------
# Conv Encoder / Decoder
# -------------------------
class ConvBlock(nn.Module):
    """Conv2d -> GroupNorm -> ReLU -> (optional Dropout2d)"""
    def __init__(self, c_in: int, c_out: int, stride_t: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out,
            kernel_size=(3, 3),
            stride=(1, stride_t),     # ONLY temporal stride
            padding=(1, 1),
        )
        self.norm = nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class DeconvBlock(nn.Module):
    """ConvTranspose2d -> GroupNorm -> ReLU -> (optional Dropout2d)"""
    def __init__(self, c_in: int, c_out: int, up_t: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        # Note: output length can differ by a few steps depending on params; we crop/pad later.
        self.deconv = nn.ConvTranspose2d(
            c_in, c_out,
            kernel_size=(3, 2 * up_t),
            stride=(1, up_t),         # ONLY temporal upsampling
            padding=(1, up_t),
            output_padding=(0, 0),
        )
        self.norm = nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.deconv(x))))


class JointAngleEncoder(nn.Module):
    """
    Input:  x_3d (B, 5, 4, 2000)
    Output: z_e  (B, D=10, 5, 200)
    """

    def __init__(self, embed_dim: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        # Treat (4) as channels, (5) as height, (T) as width
        self.stem = ConvBlock(4, 64, stride_t=1, dropout=dropout)     # 2000 -> 2000
        self.b1 = ConvBlock(64, 128, stride_t=1, dropout=dropout)     # 2000 -> 2000
        self.b2 = ConvBlock(128, 128, stride_t=2, dropout=dropout)    # 2000 -> 1000
        self.b3 = ConvBlock(128, 128, stride_t=5, dropout=dropout)    # 1000 -> 200
        self.to_d = nn.Conv2d(128, embed_dim, kernel_size=1)

    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        if x_3d.ndim != 4:
            raise ValueError(f"Expected x_3d (B, 5, 4, T), got {tuple(x_3d.shape)}")
        # (B, 5, 4, T) -> (B, 4, 5, T)
        x = x_3d.permute(0, 2, 1, 3).contiguous()
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        z_e = self.to_d(x)  # (B, D, 5, 200)
        return z_e


class JointAngleDecoder(nn.Module):
    """
    Input:  z_q  (B, D=10, 5, 200)
    Output: xhat_3d (B, 5, 4, 2000)
    """

    def __init__(self, embed_dim: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        self.from_d = nn.Conv2d(embed_dim, 128, kernel_size=1)
        # Mirror of encoder downsampling (2,5) -> upsampling (5,2)
        self.u1 = DeconvBlock(128, 128, up_t=5, dropout=dropout)      # 200 -> ~1000
        self.u2 = DeconvBlock(128, 128, up_t=2, dropout=dropout)      # ~1000 -> ~2000
        self.refine = ConvBlock(128, 64, stride_t=1, dropout=dropout)
        self.to_out = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, z_q: torch.Tensor, target_T: int = 2000) -> torch.Tensor:
        x = self.from_d(z_q)
        x = self.u1(x)
        x = self.u2(x)
        x = self.refine(x)
        x = self.to_out(x)                   # (B, 4, 5, ~T)
        x = _crop_or_pad_time(x, target_T)   # enforce exactly target_T
        # (B, 4, 5, T) -> (B, 5, 4, T)
        return x.permute(0, 2, 1, 3).contiguous()


class ChannelMixing1x1(nn.Module):
    """1×1 conv across 20 channels over time: (B,20,T)->(B,20,T)."""
    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Conv1d(20, 20, kernel_size=1, bias=True)

    def forward(self, x_20T: torch.Tensor) -> torch.Tensor:
        return self.mix(x_20T)


# -------------------------
# Full Model
# -------------------------
class JointAngleVQVAE(nn.Module):
    """
    Full VQ-VAE for joint angle tokenization.

    forward(x_3d) returns:
      recon_20T, indices, vq_loss, codebook_loss, commit_loss
    """

    def __init__(
        self,
        embed_dim: int = 10,          # D in paper
        num_codes: int = 128,         # K in paper
        commitment_cost: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = JointAngleEncoder(embed_dim=embed_dim, dropout=dropout)

        vq = VectorQuantizer(num_codes=num_codes, embed_dim=embed_dim, commitment_cost=commitment_cost)
        self.quantizer = VQ2DWrapper(vq)

        self.decoder = JointAngleDecoder(embed_dim=embed_dim, dropout=dropout)
        self.mixing = ChannelMixing1x1()

    def forward(
        self, x_3d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_3d: (B, 5, 4, 2000)
        """
        if x_3d.shape[-1] != 2000 or x_3d.shape[1] != 5 or x_3d.shape[2] != 4:
            raise ValueError(f"Expected x_3d shape (B,5,4,2000), got {tuple(x_3d.shape)}")

        z_e = self.encoder(x_3d)  # (B, 10, 5, 200)
        z_q, indices, vq_loss, codebook_loss, commit_loss = self.quantizer(z_e)

        x_hat_3d = self.decoder(z_q, target_T=x_3d.shape[-1])  # (B, 5, 4, 2000)

        # reshape to (B, 20, T)
        b, h, c, t = x_hat_3d.shape  # h=5, c=4
        recon_20T = x_hat_3d.view(b, h * c, t)

        # 1x1 mixing across channels
        recon_20T = self.mixing(recon_20T)

        return recon_20T, indices, vq_loss, codebook_loss, commit_loss


# -------------------------
# Quick sanity check (optional)
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2
    x = torch.randn(B, 5, 4, 2000)

    model = JointAngleVQVAE(embed_dim=10, num_codes=128, commitment_cost=0.25, dropout=0.0)
    recon, indices, vq_loss, codebook_loss, commit_loss = model(x)

    print("recon:", recon.shape)       # (B, 20, 2000)
    print("indices:", indices.shape)   # (B, 5, 200)
    print("vq_loss:", float(vq_loss))
    print("codebook_loss:", float(codebook_loss))
    print("commit_loss:", float(commit_loss))
