# experiments/exp2_toy_regularization.py
"""
实验2: 正则化对比（核心实验）
目标：比较 None / L2 / Dropout / EarlyStop 对频率学习的影响
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd

from models.mlp import SimpleMLP
from data.toy_data import generate_toy_data
from frequency.fft_utils import explained_variance_band
from utils.visualization import plot_flc

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 数据生成 ====================
print("Generating toy data...")
x_grid, y_grid, x_train, y_train = generate_toy_data(
    low_freq=1, high_freq=10, high_amp=0.5, n_train=400, noise=0.05
)

x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
x_grid_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1).to(device)
y_grid_t = torch.tensor(y_grid, dtype=torch.float32).unsqueeze(1).to(device)

# ==================== 正则化配置 ====================
configs = {
    'None': {'weight_decay': 0.0, 'dropout': 0.0, 'early_stop': False},
    'L2': {'weight_decay': 1e-4, 'dropout': 0.0, 'early_stop': False},
    'Dropout': {'weight_decay': 0.0, 'dropout': 0.3, 'early_stop': False},
    'EarlyStop': {'weight_decay': 0.0, 'dropout': 0.0, 'early_stop': True},
}

epochs = 200
lr = 0.01
results = {}  # {reg_name: {'ev_low': [...], 'ev_high': [...], 'test_loss': [...]}}

# ==================== 训练每个配置 ====================
for reg_name, cfg in configs.items():
    print(f"\n{'='*50}")
    print(f"Training with regularization: {reg_name}")
    print(f"{'='*50}")
    
    set_seed(42)  # 每个配置用相同种子
    
    # 创建模型（支持 dropout）
    model = SimpleMLP(width=64, depth=2, dropout=cfg['dropout']).to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9,
        weight_decay=cfg['weight_decay']
    )
    
    ev_low = []
    ev_high = []
    test_losses = []
    best_test_loss = float('inf')
    patience = 20  # 增加 patience，避免过早停止
    no_improve = 0
    
    for epoch in trange(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        out = model(x_train_t)
        loss = F.mse_loss(out, y_train_t)
        loss.backward()
        optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            pred_grid = model(x_grid_t).cpu()
            y_grid_cpu = y_grid_t.cpu()
            # 确保维度匹配
            if pred_grid.dim() == 2 and pred_grid.shape[1] == 1:
                pred_grid = pred_grid.squeeze(1)
            if y_grid_cpu.dim() == 2 and y_grid_cpu.shape[1] == 1:
                y_grid_cpu = y_grid_cpu.squeeze(1)
            test_loss = F.mse_loss(pred_grid, y_grid_cpu).item()
            test_losses.append(test_loss)
        
        # 计算频率指标
        ev1 = explained_variance_band(y_grid, pred_grid.numpy().squeeze(), k_target=1)
        ev10 = explained_variance_band(y_grid, pred_grid.numpy().squeeze(), k_target=10)
        ev_low.append(ev1)
        ev_high.append(ev10)
        
        # Early stopping
        if cfg['early_stop']:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    results[reg_name] = {
        'ev_low': ev_low,
        'ev_high': ev_high,
        'test_loss': test_losses,
        'final_epoch': len(ev_low)
    }

# ==================== 可视化 ====================
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

# 1. 绘制 FLC 对比（低频）
plt.figure(figsize=(10, 6))
for reg_name, res in results.items():
    epochs_actual = range(1, res['final_epoch'] + 1)
    plt.plot(epochs_actual, res['ev_low'], linewidth=2, label=f'{reg_name} (low freq)')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Explained Variance', fontsize=12)
plt.title('Frequency Learning Curve: Low Frequency (k=1)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/flc_low_regularization.png', dpi=200, bbox_inches='tight')
plt.close()

# 2. 绘制 FLC 对比（高频）
plt.figure(figsize=(10, 6))
for reg_name, res in results.items():
    epochs_actual = range(1, res['final_epoch'] + 1)
    plt.plot(epochs_actual, res['ev_high'], linewidth=2, label=f'{reg_name} (high freq)')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Explained Variance', fontsize=12)
plt.title('Frequency Learning Curve: High Frequency (k=10)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/flc_high_regularization.png', dpi=200, bbox_inches='tight')
plt.close()

# 3. 计算学习速度指标（AUC）
auc_data = []
for reg_name, res in results.items():
    auc_low = np.trapz(res['ev_low'], dx=1)
    auc_high = np.trapz(res['ev_high'], dx=1)
    auc_data.append({
        'Regularization': reg_name,
        'AUC_Low': auc_low,
        'AUC_High': auc_high,
        'Final_EV_Low': res['ev_low'][-1],
        'Final_EV_High': res['ev_high'][-1],
        'Final_Test_Loss': res['test_loss'][-1]
    })

df_auc = pd.DataFrame(auc_data)
df_auc.to_csv('results/data/exp2_regularization_comparison.csv', index=False)

# 4. 绘制 AUC 对比（柱状图）
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
reg_names = df_auc['Regularization'].values
axes[0].bar(reg_names, df_auc['AUC_Low'], alpha=0.7, color='blue')
axes[0].set_ylabel('AUC (Low Freq)', fontsize=11)
axes[0].set_title('Learning Speed: Low Frequency', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(reg_names, df_auc['AUC_High'], alpha=0.7, color='red')
axes[1].set_ylabel('AUC (High Freq)', fontsize=11)
axes[1].set_title('Learning Speed: High Frequency', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/auc_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

print("\n✅ Experiment 2 completed!")
print("Results saved to:")
print("  - results/figures/flc_low_regularization.png")
print("  - results/figures/flc_high_regularization.png")
print("  - results/figures/auc_comparison.png")
print("  - results/data/exp2_regularization_comparison.csv")
print("\n观察：正则化应该延缓高频学习，但低频受影响较小！")

