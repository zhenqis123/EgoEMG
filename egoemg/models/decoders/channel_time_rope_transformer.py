import math
import torch
from torch import nn
from typing import Optional, List, Literal

def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """标准 RoPE 旋转应用逻辑"""
    # x: (..., dim), cos/sin: (..., dim//2)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # 根据 RoPE 公式进行旋转复数运算
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2)

class CyRoPETransformerEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        num_channels: int = 16,
        channel_angles: Optional[List[float]] = None,
        time_base: float = 10000.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert model_dim % 4 == 0, "Embedding dim 必须能被 4 整除 (拆分 d/2 后需为偶数)"
        
        self.d = model_dim
        self.half_d = model_dim // 2
        self.C = num_channels
        self.time_base = time_base
        
        # 1. 空间/通道角度配置
        if channel_angles is not None:
            if len(channel_angles) != num_channels:
                raise ValueError(f"channel_angles 长度必须为 {num_channels}")
            # 使用输入的弧度值
            self.register_buffer("spatial_pos", torch.tensor(channel_angles, dtype=torch.float32))
        else:
            # 默认平分：每个通道间隔 pi/8 (即 2*pi/16)
            default_angles = torch.arange(num_channels).float() * (2.0 * math.pi / num_channels)
            self.register_buffer("spatial_pos", default_angles)

        # 2. Transformer 结构
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _get_time_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """时间维度频率计算：符合标准 RoPE"""
        # i 从 0 到 d/4-1 (对应论文中的 d/2 子空间)
        inv_freq = 1.0 / (self.time_base ** (torch.arange(0, self.half_d, 2, device=device).float() / self.half_d))
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        return freqs.cos(), freqs.sin()

    def _get_spatial_cos_sin(self, device: torch.device, dtype: torch.dtype):
        """空间维度频率计算：严格对齐论文公式 (4)"""
        # omega_0 = 2*pi / C
        omega_0 = (2.0 * math.pi) / self.C
        
        # 论文公式 (4): theta_c^(i) = omega_0 ^ (2i / (d/2))，其中 i 从 1 到 d/4
        # 为了让输入角度作为锚点，我们需要缩放频率，使得在最高维索引处对应的 theta 作用效果为 1
        # 索引 idx = [1, 2, ..., d/4]
        idx = torch.arange(1, (self.half_d // 2) + 1, device=device).float()
        
        # 计算多尺度空间频率系数
        # 注意：这里将 omega_0 作为基底，实现多尺度感知
        spatial_freqs = (omega_0 ** (2 * idx / self.half_d)) / omega_0  # 除以 omega_0 确保锚点处缩放为 1
        
        # 使用配置的角度位置 (B, C) 或 (C)
        freqs = torch.einsum("i,j->ij", self.spatial_pos.to(dtype), spatial_freqs)
        return freqs.cos(), freqs.sin()

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T, D) -> Batch, Channels, Time, EmbeddingDim
        """
        B, C, T, D = x.shape
        device, dtype = x.device, x.dtype
        
        # 1. 拆分维度
        z_t = x[..., :self.half_d] # 时间半区
        z_c = x[..., self.half_d:] # 空间半区
        
        # 2. 应用时间 RoPE (在 T 维度旋转)
        cos_t, sin_t = self._get_time_cos_sin(T, device, dtype) # (T, half_d//2)
        # 调整形状以匹配 (B, C, T, half_d)
        z_t = _apply_rotary_emb(z_t, cos_t.unsqueeze(0).unsqueeze(0), sin_t.unsqueeze(0).unsqueeze(0))
        
        # 3. 应用空间 CyRoPE (在 C 维度旋转)
        cos_c, sin_c = self._get_spatial_cos_sin(device, dtype) # (C, half_d//2)
        # 调整形状以匹配 (B, C, T, half_d)
        z_c = _apply_rotary_emb(z_c, cos_c.unsqueeze(0).unsqueeze(2), sin_c.unsqueeze(0).unsqueeze(2))
        
        # 4. 合并并展平为 Transformer Token 序列
        z_combined = torch.cat([z_t, z_c], dim=-1) # (B, C, T, D)
        # 展平为 (B, C*T, D)
        tokens = z_combined.reshape(B, C * T, D)
        
        # 5. Transformer 运算
        output = self.transformer(tokens)
        
        # 6. 还原形状回 (B, C, T, D)
        return output.view(B, C, T, D)