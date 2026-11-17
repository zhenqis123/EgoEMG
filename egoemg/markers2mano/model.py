"""The global-pose-head graph transformer, shared by conversion and training."""

import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs
import networkx as nx



def six_d_to_rot_matrix(d6: torch.Tensor) -> torch.Tensor:
    """将6D连续表示转换为3x3旋转矩阵。"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = nn.functional.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def get_laplacian_eigenvectors(edge_index, num_nodes, k=8):
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adj[u, v] = 1
        adj[v, u] = 1
    deg = np.diag(adj.sum(axis=1))
    laplacian = deg - adj
    _, eigenvectors = eigs(laplacian.astype(float), k=k, which='SR', tol=1e-2)
    return torch.from_numpy(eigenvectors.real).float()


def get_shortest_path_distances(edge_index, num_nodes):
    graph = nx.Graph()
    for i in range(edge_index.shape[1]):
        graph.add_edge(edge_index[0, i].item(), edge_index[1, i].item())
    path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    distances = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    max_dist = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist = path_lengths[i].get(j, 0)
            distances[i, j] = dist
            if dist > max_dist:
                max_dist = dist
    return distances, max_dist + 1


class GraphTransformerLayer4x(nn.Module):
    """Graph attention with fourfold feed-forward expansion."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, ffn_ratio=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        B, N, _ = x.shape
        num_heads = self.attention.num_heads

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(0).repeat(B, 1, 1, 1)
            attn_bias = attn_bias.view(B * num_heads, N, N)

        attn_output, _ = self.attention(x, x, x, attn_mask=attn_bias)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class EfficientGraphTransformer4x(nn.Module):
    """
    The markers2mano architecture: structural encodings and one global 16-joint pose head.
    """

    def __init__(self, num_markers=21, beta_dim=10, embed_dim=256,
                 num_layers=6, num_heads=8, dropout=0.1, lap_eig_k=8, ffn_ratio=4):
        super().__init__()
        self.num_markers = num_markers
        pose_dim_6d = 16 * 6

        # --- 1. Graph structure (fixed) ---
        edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                 [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20]]
        undirected_edges = edges + [[v, u] for u, v in edges]
        edge_index = torch.LongTensor(undirected_edges).t().contiguous()

        # --- 2. Structural encoding (computed once) ---
        lap_eigvecs = get_laplacian_eigenvectors(edge_index, num_markers, k=lap_eig_k)
        self.register_buffer('lap_eigvecs', lap_eigvecs)
        self.lap_embedding = nn.Linear(lap_eig_k, embed_dim)

        spd, max_dist = get_shortest_path_distances(edge_index, num_markers)
        self.register_buffer('spd', spd)
        self.spd_embedding = nn.Embedding(max_dist, num_heads)

        # --- 3. Network layers (4x FFN) ---
        self.input_embedding = nn.Linear(3, embed_dim)
        self.transformer_layers = nn.ModuleList(
            [GraphTransformerLayer4x(embed_dim, num_heads, dropout, ffn_ratio=ffn_ratio)
             for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # --- 4. Prediction heads ---
        self.beta_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, beta_dim),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(num_markers * embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, pose_dim_6d),
        )

    def forward(self, markers):
        attn_bias = self.spd_embedding(self.spd).permute(2, 0, 1)
        lap_feat = self.lap_embedding(self.lap_eigvecs)
        x = self.input_embedding(markers) + lap_feat

        for layer in self.transformer_layers:
            x = layer(x, attn_bias=attn_bias)

        x = self.final_norm(x)

        global_feat = x.mean(dim=1)
        pred_beta = self.beta_head(global_feat)

        local_feats_flat = x.flatten(start_dim=1)
        pred_pose_d = self.pose_head(local_feats_flat)

        return pred_pose_d, pred_beta
