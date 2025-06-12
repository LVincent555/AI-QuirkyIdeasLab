# --- main.py ---
import torch
# 导入我们之前定义的所有类
from mai_gem_agent import MAI_GEM_Agent
from trainer import MultiScaleTrainer

def main():
    # --- 1. 定义模型超参数 ---
    print("--- Initializing MAI-GEM Flagship Model ---")
    INPUT_DIM = 16
    HIDDEN_DIM = 64
    STATE_DIM = 32
    CHEM_DIM = 8
    PHYS_DIM = 4
    SELF_MODEL_HIDDEN_DIM = 32
    NUM_HEADS = 4
    
    # 学习率也体现了时间尺度
    learning_rates = {
        'fast': 1e-3,
        'meta': 5e-4,
        'self': 5e-4,
        'struc': 1e-5, # 结构学习最慢
    }
    
    # 损失权重定义了智能体的“价值观”
    loss_weights = {
        'world': 1.0,   # 预测外部世界很重要
        'homeo': 2.0,   # 维持物理生存更重要
        'self': 0.5,    # 自我反思是次要任务
    }

    # --- 2. 实例化智能体和训练器 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = MAI_GEM_Agent(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, state_dim=STATE_DIM,
        chem_dim=CHEM_DIM, phys_dim=PHYS_DIM, self_model_hidden_dim=SELF_MODEL_HIDDEN_DIM,
        num_heads=NUM_HEADS
    ).to(device)
    
    trainer = MultiScaleTrainer(agent, learning_rates, loss_weights)
    print(f"Model and Trainer initialized on {device}.")

    # --- 3. 模拟训练循环 ---
    print("\n--- Starting Simulation Loop ---")
    NUM_EPOCHS = 10
    SEQ_LEN = 50
    BATCH_SIZE = 4
    INTERNAL_HISTORY_LEN = 10 # 自我模型回顾的步数

    for epoch in range(NUM_EPOCHS):
        # --- 生成模拟数据 ---
        # 外部感官序列
        x_seq = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_DIM).to(device)
        # 内部物理状态 (e.g., 饥饿, 疲劳, 体温, 损伤)
        s_phys = torch.rand(BATCH_SIZE, PHYS_DIM).to(device) * 2 - 1 # 随机偏离稳态
        # 物理稳态目标 (通常是全零)
        target_s_phys = torch.zeros_like(s_phys)
        # 内部状态历史 (用于自我模型)
        internal_state_history = torch.randn(BATCH_SIZE, INTERNAL_HISTORY_LEN, STATE_DIM + CHEM_DIM).to(device)

        # --- 执行一个训练步骤 ---
        losses = trainer.train_step(x_seq, s_phys, target_s_phys, internal_state_history)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Losses -> World: {losses['world_prediction']:.4f} | "
              f"Homeostatic: {losses['homeostatic']:.4f} | "
              f"Self-Model: {losses['self_model']:.4f}")
        
        # 我们可以检查智能体内部状态的变化
        if agent.S_chem is not None:
            print(f"  Agent S_chem (mean): {agent.S_chem.mean().item():.3f}")
        
        # 我们可以检查结构的变化
        s_struc_entropy = torch.distributions.Categorical(probs=agent.structural_core.S_struc[0]).entropy().mean()
        print(f"  Agent S_struc entropy (mean): {s_struc_entropy.item():.3f}")

if __name__ == '__main__':
    main()




# 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
# 作者：vincent  
# 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  