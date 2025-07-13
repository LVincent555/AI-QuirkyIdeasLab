# --- self_model_core.py ---
import torch
import torch.nn as nn

class SelfModelCore(nn.Module):
    """
    MAI-GEM的自我模型核心 (高阶进程)。
    观察自身内部状态，并预测其未来演变。
    """
    def __init__(self, state_dim, chem_dim, self_model_hidden_dim):
        super().__init__()
        # 一个循环网络（如GRU）来模拟和预测状态序列
        self.predictor = nn.GRU(input_size=state_dim + chem_dim, 
                                hidden_size=self_model_hidden_dim, 
                                batch_first=True)
        self.output_proj = nn.Linear(self_model_hidden_dim, state_dim + chem_dim)

    def forward(self, internal_state_sequence: torch.Tensor):
        """
        Args:
            internal_state_sequence (Tensor): 智能体过去K个时间步的内部状态序列 (B, K, S+C)。
        
        Returns:
            Tensor: 对下一个时间步内部状态的预测 (B, S+C)。
        """
        # 使用GRU处理历史状态序列
        gru_out, _ = self.predictor(internal_state_sequence)
        # 取最后一个时间步的输出来做预测
        last_step_out = gru_out[:, -1, :]
        predicted_next_state = self.output_proj(last_step_out)
        return predicted_next_state

    def compute_self_model_loss(self, predicted_state: torch.Tensor, actual_next_state: torch.Tensor):
        """
        计算自我模型损失，即自我预测误差。
        """
        return nn.MSELoss()(predicted_state, actual_next_state)
    
    
# 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
# 作者：vincent  
# 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  