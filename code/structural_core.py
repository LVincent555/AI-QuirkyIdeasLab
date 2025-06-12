# --- structural_core.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuralCore(nn.Module):
    """
    MAI-GEM的结构核心 (慢进程)。
    实现了一个由S_struc参数化的稀疏注意力机制。
    """
    def __init__(self, hidden_dim, num_heads, top_k_ratio=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.top_k_ratio = top_k_ratio

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # S_struc: 结构状态，现在是一个可学习的参数，代表注意力图的先验
        # 初始化为均匀分布，代表无先验知识
        self.S_struc = nn.Parameter(torch.ones(num_heads, hidden_dim, hidden_dim) / hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): 输入序列，形状为 (L, B, D)。
        
        Returns:
            Tensor: 输出序列，形状为 (L, B, D)。
        """
        L, B, D = x.shape
        
        # 1. 投影到 Q, K, V
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        # 2. 调整形状以适应多头注意力
        q = q.view(L, B, self.num_heads, self.head_dim).transpose(0, 2) # (H, B, L, D_h)
        k = k.view(L, B, self.num_heads, self.head_dim).transpose(0, 2) # (H, B, L, D_h)
        v = v.view(L, B, self.num_heads, self.head_dim).transpose(0, 2) # (H, B, L, D_h)

        # 3. 计算原始注意力分数
        attn_scores = torch.einsum('hblf,hbtf->hblt', q, k) / (self.head_dim ** 0.5)

        # 4. 引入S_struc作为注意力先验 (核心机制)
        # S_struc为注意力矩阵增加了一个可学习的偏置，引导注意力模式
        # 这比硬性门控更灵活，更符合生物可塑性
        structural_bias = torch.log(self.S_struc + 1e-8) # Log-space for stability
        attn_scores = attn_scores + structural_bias[:, :L, :L] # 截取以匹配序列长度

        # 5. 实现稀疏性 (Top-K)
        # 为了效率，我们只允许每个查询关注得分最高的K个键
        k = int(self.top_k_ratio * L)
        top_k_indices = torch.topk(attn_scores, k, dim=-1).indices
        
        # 创建一个稀疏掩码
        mask = torch.full_like(attn_scores, float('-inf'))
        mask.scatter_(-1, top_k_indices, 0.0)
        
        # 应用掩码和softmax
        attn_weights = F.softmax(attn_scores + mask, dim=-1)

        # 6. 计算输出
        attn_output = torch.einsum('hblt,hbtv->hblv', attn_weights, v)
        attn_output = attn_output.transpose(0, 2).contiguous().view(L, B, D)
        
        return self.out_proj(attn_output)



# 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
# 作者：vincent  
# 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  