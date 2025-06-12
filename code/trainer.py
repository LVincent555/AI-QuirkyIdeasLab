# --- trainer.py ---
import torch
import torch.optim as optim
from collections import defaultdict

class MultiScaleTrainer:
    """
    一个原则性的训练器，用于训练MAI-GEM智能体。
    它管理多个优化器，并根据理论实现多尺度学习。
    """
    def __init__(self, agent: 'MAI_GEM_Agent', learning_rates: dict, loss_weights: dict):
        self.agent = agent
        self.loss_weights = loss_weights

        # --- 为不同时间尺度的参数创建独立的优化器 ---
        # 快进程 (φ_fast): 负责处理瞬时数据流
        phi_fast = list(agent.temporal_core.parameters()) + \
                   list(agent.input_proj.parameters()) + \
                   list(agent.output_proj.parameters())
        self.optimizer_fast = optim.Adam(phi_fast, lr=learning_rates['fast'])

        # 中进程 (φ_meta): 负责情境适应和代谢调节
        phi_meta = list(agent.metabolic_core.parameters())
        self.optimizer_meta = optim.Adam(phi_meta, lr=learning_rates['meta'])

        # 慢进程 (φ_struc): 负责固化长期知识
        phi_struc = list(agent.structural_core.parameters())
        self.optimizer_struc = optim.Adam(phi_struc, lr=learning_rates['struc'])

        # 高阶进程 (φ_self): 负责自我建模
        phi_self = list(agent.self_model_core.parameters())
        self.optimizer_self = optim.Adam(phi_self, lr=learning_rates['self'])

        self.optimizers = {
            'fast': self.optimizer_fast,
            'meta': self.optimizer_meta,
            'struc': self.optimizer_struc,
            'self': self.optimizer_self
        }
        
        self.history = defaultdict(list)

    def train_step(self, x_seq, s_phys, target_s_phys, internal_state_history):
        """
        执行一个完整的训练步骤。
        """
        # --- 1. 前向传播，获取所有损失 ---
        self.agent.train()
        y_pred, losses = self.agent(x_seq, s_phys, target_s_phys, internal_state_history)

        # --- 2. 计算加权总损失 ---
        # 每个损失都驱动不同参数的更新
        # F_total = w1*F_ext + w2*F_homeo + w3*F_self
        # 理论上，结构损失应来自更长期的平均，这里简化
        
        # 损失驱动快速和中等进程
        loss_fast_meta = self.loss_weights['world'] * losses['world_prediction'] + \
                         self.loss_weights['homeo'] * losses['homeostatic']
        
        # 损失驱动自我模型进程
        loss_self = self.loss_weights['self'] * losses['self_model']

        # 结构损失应基于长期表现，这里我们用一个代理：
        # 一个好的结构应该能最小化所有其他损失
        loss_struc = loss_fast_meta.detach() + loss_self.detach()

        # --- 3. 多尺度反向传播与更新 ---
        # a. 更新快进程和中进程参数
        self.optimizer_fast.zero_grad()
        self.optimizer_meta.zero_grad()
        loss_fast_meta.backward(retain_graph=True) # retain_graph因为损失被复用
        self.optimizer_fast.step()
        self.optimizer_meta.step()

        # b. 更新自我模型参数
        self.optimizer_self.zero_grad()
        loss_self.backward(retain_graph=True)
        self.optimizer_self.step()

        # c. 更新结构参数 (最慢)
        self.optimizer_struc.zero_grad()
        loss_struc.backward()
        self.optimizer_struc.step()
        
        # --- 4. 记录历史 ---
        for k, v in losses.items():
            self.history[k].append(v.item())
        
        return {k: v.item() for k, v in losses.items()}



# 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
# 作者：vincent  
# 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  