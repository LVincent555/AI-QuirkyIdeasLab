# --- metabolic_core.py ---
import torch
import torch.nn as nn

class MetabolicCore(nn.Module):
    """
    MAI-GEM的代谢/化学核心 (中进程)。
    根据外部意外 (surprise) 和内部物理需求 (homeostatic error) 生成化学状态 S_chem。
    """
    def __init__(self, chem_dim, phys_dim):
        super().__init__()
        self.chem_dim = chem_dim
        self.phys_dim = phys_dim

        # 将物理误差和意外信号映射到化学状态空间
        self.phys_proj = nn.Linear(phys_dim, chem_dim)
        self.surprise_proj = nn.Linear(1, chem_dim)
        
        # 维持一个慢变的化学状态基线
        self.chem_baseline = nn.Parameter(torch.zeros(1, chem_dim))
        self.chem_decay = 0.99 # 化学状态会缓慢回归基线

    def forward(self, surprise: torch.Tensor, homeostatic_error: torch.Tensor):
        """
        Args:
            surprise (Tensor): 标量，代表外部世界的预测误差。
            homeostatic_error (Tensor): 向量 (B, P)，代表物理状态偏离稳态的程度。
        
        Returns:
            Tensor: 新的化学状态 S_chem (B, C)。
        """
        # 物理需求和外部意外驱动化学状态的变化
        chem_drive_from_phys = self.phys_proj(homeostatic_error)
        chem_drive_from_surprise = self.surprise_proj(surprise.unsqueeze(-1))
        
        # 新的化学状态是基线、驱动和衰减的组合
        # (这里我们简化，不使用递归状态，直接生成)
        s_chem = torch.tanh(self.chem_baseline + chem_drive_from_phys + chem_drive_from_surprise)
        
        return s_chem

    def compute_homeostatic_loss(self, s_phys: torch.Tensor, target_phys_state: torch.Tensor):
        """
        计算体内平衡损失，这是智能体最基本的内在驱动力。
        """
        return nn.MSELoss()(s_phys, target_phys_state)



# 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
# 作者：vincent  
# 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  