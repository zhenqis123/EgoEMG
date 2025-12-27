from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from emg2pose.models.quantizers.vq import GroupedVectorQuantizer


class ResidualMLPBlock(nn.Module):
    """深层 MLP 的基本残差单元"""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接：帮助深层网络收敛
        return self.relu(x + self.net(x))


class HeavyMLPEncoder(nn.Module):
    """通过堆叠残差块实现的高容量 Encoder"""
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        hidden_dim: int = 512, 
        num_blocks: int = 8, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. 初始投影：从 20 维扩展到高维
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 2. 堆叠残差块：这是主要的参数来源
        self.res_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        
        # 3. 最终投影：映射到 VQ 的 embed_dim (50)
        self.last_layer = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.res_blocks(x)
        x = self.last_layer(x)
        return x


class TimeStepGroupedVQVAE(nn.Module):
    """使用深层残差 MLP 实现的单步量化 VQ-VAE"""
    input_mode = "bct"

    def __init__(
        self,
        input_dim: int = 20,
        embed_dim: int = 50,
        num_groups: int = 5,
        group_dim: int = 10,
        num_codes: int = 128,
        commitment_cost: float = 0.25,
        dropout: float = 0.1,
        downsample_rate: float | None = None,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        # 新增超参数：控制 Encoder 深度和宽度
        enc_hidden_dim: int = 256,
        enc_num_blocks: int = 12,  # 堆叠层数
    ) -> None:
        super().__init__()
        
        if embed_dim != num_groups * group_dim:
            raise ValueError(f"embed_dim ({embed_dim}) 必须等于 num_groups * group_dim")

        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.num_groups = int(num_groups)
        self.group_dim = int(group_dim)
        self.num_codes = int(num_codes)
        if downsample_rate is None:
            self.downsample_rate = None
        else:
            rate = float(downsample_rate)
            if rate <= 0:
                raise ValueError("downsample_rate must be > 0.")
            self.downsample_rate = rate

        # --- 重型 Encoder ---
        self.encoder = HeavyMLPEncoder(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            hidden_dim=enc_hidden_dim,
            num_blocks=enc_num_blocks,
            dropout=dropout
        )

        # --- 量化器 ---
        self.quantizer = GroupedVectorQuantizer(
            num_groups=self.num_groups,
            num_codes=self.num_codes,
            group_dim=self.group_dim,
            commitment_cost=commitment_cost,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
        )

        # --- 轻型 Decoder (保持你之前的配置) ---
        hidden_dim = 256
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.input_dim)
        )

    def _resample_time(self, x: torch.Tensor, target_t: int) -> torch.Tensor:
        if x.shape[-1] == target_t:
            return x
        return F.interpolate(x, size=target_t, mode="linear", align_corners=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, C, T), got {tuple(x.shape)}")
        b, c, t = x.shape
        if c != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {c}")

        original_t = t
        if self.downsample_rate is not None and self.downsample_rate != 1.0:
            target_t = max(1, int(round(t * self.downsample_rate)))
            x = self._resample_time(x, target_t)
            t = x.shape[-1]
        
        # 1. 展平时间步，确保单步独立处理 (B*T, C)
        x_t = x.transpose(1, 2).contiguous() 
        flat = x_t.view(b * t, c) 

        # 2. 前向传播
        z_e = self.encoder(flat) 
        z_q, indices, vq_loss, codebook_loss, commit_loss = self.quantizer(z_e)
        recon = self.decoder(z_q) 

        # 3. 恢复形状
        recon = recon.view(b, t, c).transpose(1, 2).contiguous()
        if recon.shape[-1] != original_t:
            recon = self._resample_time(recon, original_t)
        indices = indices.view(b, t, self.num_groups).permute(0, 2, 1).contiguous()
        return recon, indices, vq_loss, codebook_loss, commit_loss
