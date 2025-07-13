# --- mai_gem_agent.py ---
import torch
import torch.nn as nn
# 导入我们之前定义的所有核心模块
from temporal_core import TemporalCore
from structural_core import StructuralCore
from metabolic_core import MetabolicCore
from self_model_core import SelfModelCore

class MAI_GEM_Agent(nn.Module):
    """
    旗舰级MAI-GEM智能体。
    将所有核心模块组装在一起，实现多尺度、多目标的学习。
    """
    def __init__(self, input_dim, hidden_dim, state_dim, chem_dim, phys_dim, 
                 self_model_hidden_dim, num_heads=8):
        super().__init__()
        
        # --- 实例化所有核心模块 ---
        self.temporal_core = TemporalCore(hidden_dim, state_dim)
        self.structural_core = StructuralCore(hidden_dim, num_heads)
        self.metabolic_core = MetabolicCore(chem_dim, phys_dim)
        self.self_model_core = SelfModelCore(state_dim, chem_dim, self_model_hidden_dim)

        # --- 投影层 ---
        # 将输入投影到模型的隐藏维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 将模型的输出投影回输入空间，用于计算预测误差
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # --- 内部状态变量 ---
        # 这些变量在forward pass中被动态更新
        self.S_chem = None
        self.S_phys = None
        
    def forward(self, x_seq, s_phys, target_s_phys, internal_state_history):
        """
        一个完整的、多层次的前向传播过程。
        """
        # 1. 计算物理/代谢需求
        homeostatic_error = self.metabolic_core.compute_homeostatic_loss(s_phys, target_s_phys)
        
        # 2. 预测外部世界 (F_ext)
        # 输入 -> 结构核心 (注意力) -> 时间核心 (SSM) -> 输出
        x_proj = self.input_proj(x_seq)
        x_structured = self.structural_core(x_proj)
        x_temporal = self.temporal_core(x_structured)
        y_pred = self.output_proj(x_temporal)
        
        # 计算外部世界预测误差 (Surprise)
        world_prediction_error = nn.MSELoss()(y_pred, x_seq)

        # 3. 更新化学状态 S_chem
        self.S_chem = self.metabolic_core(world_prediction_error.detach(), homeostatic_error.detach())

        # 4. 预测内部世界 (F_self)
        # (注意：这里的实现有简化，实际应传入历史的S_temp和S_chem)
        predicted_next_internal_state = self.self_model_core(internal_state_history)
        
        # 5. 组装所有损失，用于多目标学习
        losses = {
            'world_prediction': world_prediction_error,
            'homeostatic': homeostatic_error,
            'self_model': self.self_model_core.compute_self_model_loss(
                predicted_next_internal_state, 
                torch.cat([torch.randn_like(predicted_next_internal_state[:, :self.temporal_core.state_dim]), self.S_chem], dim=1) # 伪造的actual_next_state
            )
        }
        
        return y_pred, losses




# 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
# 作者：vincent  
# 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  