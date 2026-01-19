import torch
import torch.nn as nn
import torch.nn.functional as F

class DeviceAwareMoE(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_ffn, 
                 num_experts=4, 
                 num_devices=5, 
                 device_emb_dim=16, 
                 top_k=1, 
                 capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # 1. 专家网络 (Experts)
        # 使用 ModuleList 存储多个简单的 FFN
        # 针对 FastConformer，专家通常是一个两层 MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.SiLU(), 
                nn.Dropout(0.1),
                nn.Linear(d_ffn, d_model),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
        # 2. 路由器 (Router) - 支持 Device ID
        self.device_emb = nn.Embedding(num_devices, device_emb_dim)
        # 门控网络输入 = 信号特征 + 设备特征
        self.gate = nn.Linear(d_model + device_emb_dim, num_experts)
        
    def forward(self, x, device_ids):
        """
        x: [Batch, Time, d_model]
        device_ids: [Batch] or [Batch, Time]
        """
        batch_size, time_steps, _ = x.shape
        
        # --- A. 路由计算 ---
        # 扩展 device_ids 以匹配时间步 (如果传入的是 [Batch])
        if device_ids.dim() == 1:
            d_emb = self.device_emb(device_ids).unsqueeze(1).expand(-1, time_steps, -1)
        else:
            d_emb = self.device_emb(device_ids)
            
        # 拼接特征用于路由
        router_input = torch.cat([x, d_emb], dim=-1)
        
        # 计算 Logits [B, T, Num_Experts]
        router_logits = self.gate(router_input)
        
        # --- B. 负载均衡损失 (Auxiliary Loss) ---
        # 即使只用自己实现的 MoE，也必须加这个，否则模型会偷懒只用 1 个专家
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 1. 密度 (每个专家被选中的概率)
        density = router_probs.mean(dim=(0, 1)) # [Num_Experts]
        # 2. 密度代理 (Logits 均值，避免不可导)
        density_proxy = router_logits.mean(dim=(0, 1))
        
        # 简单的负载均衡 Loss: 期望概率分布尽可能均匀
        # 使用方差/变异系数，或者简单的 Mean Square Error (target=1/N)
        # 这里使用最通用的 "Switch Transformer" 风格的 aux loss
        # loss = Num_Experts * sum(density * density_proxy)
        aux_loss = (self.num_experts * (density * F.softmax(density_proxy, dim=-1)).sum())
        
        # --- C. 选择与加权 (Masked Strategy) ---
        # 选出 Top-K
        # weights: [B, T, k], indices: [B, T, k]
        routing_weights, selected_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 归一化权重 (让选中的 k 个权重和为 1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # --- D. 专家计算 (Dense Mode) ---
        # 这种写法逻辑最简单：让所有专家都算一遍
        # 只要 expert 数量 < 8，这种写法在 PyTorch 里效率很高
        
        # Stack inputs -> [Experts, B, T, D] -> 实际上不需要复制，只需循环调用
        # 更好的方式是并行计算：将所有专家的权重合并为一个大矩阵 (Grouped Linear)
        # 但为了方便，我们直接循环：
        
        final_output = torch.zeros_like(x)
        
        # 创建 One-hot Mask [B, T, Num_Experts]
        # 将 selected_indices 转为 one-hot
        expert_mask = F.one_hot(selected_indices, num_classes=self.num_experts).sum(dim=2)
        
        # 遍历所有专家 (Python 循环 4-8 次是非常快的)
        for i, expert in enumerate(self.experts):
            # 1. 计算专家输出
            expert_out = expert(x) # [B, T, D]
            
            # 2. 检查当前专家是否被该 token 选中
            # mask_i: [B, T] (0 或 1)
            mask_i = expert_mask[:, :, i]
            
            # 3. 如果没有任何 token 选中该专家，跳过加权计算 (省一点点时间)
            if mask_i.sum() == 0:
                continue
                
            # 4. 获取对应的路由权重
            # 我们需要知道这个专家是该 token 的第几顺位 (top1 还是 top2)
            # 这是一个难点。为了简化，我们使用 "Soft Routing" 的加权求和技巧：
            # 直接用 router_probs (全概率) 或者重新 gather
            
            # 简易方案：只加权 Top-K 的部分
            # 找到 routing_weights 中对应专家 i 的权重
            # 这是一个 trick: 使用 scatter 构建全尺寸权重图
            
            pass 
        
        # --- 优化 D 部分：更简单的加权逻辑 ---
        # 上面的循环逻辑处理权重有点麻烦，我们换一种更向量化的写法：
        
        # 1. 运行所有专家
        # expert_outputs: [B, T, Num_Experts, D]
        # 这一步可以通过 torch.vmap (PyTorch 2.0+) 优化，或者简单堆叠
        expert_results = [e(x) for e in self.experts] 
        expert_results = torch.stack(expert_results, dim=2) 
        
        # 2. 构建加权矩阵
        # router_probs: [B, T, E]
        # 我们只保留 top-k 的概率，其余置 0
        mask = torch.zeros_like(router_probs).scatter_(2, selected_indices, 1.0)
        masked_probs = router_probs * mask
        
        # 再次归一化，保证和为 1
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 3. Einsum 加权求和
        # [B, T, E] * [B, T, E, D] -> [B, T, D]
        final_output = torch.einsum('bte, bted -> btd', masked_probs, expert_results)
        
        return final_output, aux_loss