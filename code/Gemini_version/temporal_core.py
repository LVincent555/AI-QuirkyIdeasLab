# --- temporal_core.py ---
import torch
import torch.nn as nn
import maigem_core # 导入我们编译的C++模块

class TemporalCore(nn.Module):
    """
    MAI-GEM的时间核心 (快进程)。
    实现了一个受S_chem调制的选择性状态空间模型 (SSM)。
    """
    def __init__(self, hidden_dim, state_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # 基础状态矩阵 A 和 B (可学习)
        self.A = nn.Parameter(torch.randn(hidden_dim, state_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim, state_dim))
        
        # 投影层 C 和 D (可学习)
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim))
        self.D = nn.Parameter(torch.randn(1, hidden_dim)) # Direct path from input to output

        # 用于生成选择性参数 delta 的网络
        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq (Tensor): 输入序列，形状为 (L, B, D)，L=序列长度, B=批大小, D=隐藏维度。
        
        Returns:
            Tensor: 输出序列，形状为 (L, B, D)。
        """
        L, B, D = x_seq.shape
        
        # 1. 生成选择性参数 delta
        delta_seq = torch.log(1 + torch.exp(self.delta_proj(x_seq))) # Softplus确保delta为正

        # 2. 调用C++核心执行选择性扫描
        # 注意：我们的C++代码目前只支持批大小为1，为了简化，我们这里循环处理
        # 在生产环境中，C++代码也应该被并行化以处理批次
        y_scan_list = []
        for i in range(B):
            x_np = x_seq[:, i, :].detach().cpu().numpy().astype('float64')
            delta_np = delta_seq[:, i, :].detach().cpu().numpy().astype('float64')
            A_np = self.A.detach().cpu().numpy().astype('float64')
            B_np = self.B.detach().cpu().numpy().astype('float64')
            
            scan_result_np = maigem_core.selective_scan(x_np, delta_np, A_np, B_np)
            y_scan_list.append(torch.from_numpy(scan_result_np).to(x_seq.device).float())
        
        y_scan = torch.stack(y_scan_list, dim=1)

        # 3. 通过投影层C和D计算最终输出
        # y_t = C * y_scan_t + D * x_t
        y_out = torch.einsum('lbd,dn->lbn', y_scan, self.C) + x_seq * self.D
        
        return y_out
    
    
    
# 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
# 作者：vincent  
# 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  