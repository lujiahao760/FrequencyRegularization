# experiments/exp3_frc_analysis.py
"""
实验3：FRC 指标与泛化性能的关系
核心创新：验证 FRC 作为泛化性能的预测指标

假设：FRC 越高 → 高频能量越高 → 过拟合风险越大 → 测试误差越大
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from frequency.frc import compute_frc, compute_frc_trajectory
from utils.visualization import plot_flc

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 数据生成 ====================
print("Generating toy data with noise...")
x_grid, y_grid, x_train, y_train = generate_toy_data(
    low_freq=1, high_freq=10, high_amp=0.5, n_train=400, noise=0.05
)

x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
x_grid_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1).to(device)
y_grid_t = torch.tensor(y_grid, dtype=torch.float32).unsqueeze(1).to(device)

# ==================== 不同正则化配置 ====================
configs = {
    'None': {'weight_decay': 0.0, 'dropout': 0.0, 'early_stop': False},
    'L2_weak': {'weight_decay': 1e-5, 'dropout': 0.0, 'early_stop': False},
    'L2_medium': {'weight_decay': 1e-4, 'dropout': 0.0, 'early_stop': False},
    'L2_strong': {'weight_decay': 1e-3, 'dropout': 0.0, 'early_stop': False},
    'Dropout_weak': {'weight_decay': 0.0, 'dropout': 0.1, 'early_stop': False},
    'Dropout_medium': {'weight_decay': 0.0, 'dropout': 0.3, 'early_stop': False},
    'Dropout_strong': {'weight_decay': 0.0, 'dropout': 0.5, 'early_stop': False},
}

epochs = 200
lr = 0.01
results = []

# ==================== 训练每个配置 ====================
for reg_name, cfg in configs.items():
    print(f"\n{'='*50}")
    print(f"Training: {reg_name}")
    print(f"{'='*50}")
    
    set_seed(42)
    
    model = SimpleMLP(width=64, depth=2, dropout=cfg['dropout']).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9,
        weight_decay=cfg['weight_decay']
    )
    
    frc_trajectory = []
    test_losses = []
    predictions_history = []
    
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
            if pred_grid.dim() == 2 and pred_grid.shape[1] == 1:
                pred_grid = pred_grid.squeeze(1)
            if y_grid_cpu.dim() == 2 and y_grid_cpu.shape[1] == 1:
                y_grid_cpu = y_grid_cpu.squeeze(1)
            test_loss = F.mse_loss(pred_grid, y_grid_cpu).item()
            test_losses.append(test_loss)
        
        # 计算 FRC
        pred_np = pred_grid.numpy().squeeze()
        frc, e_low, e_high = compute_frc(pred_np, low_freq_range=(1, 3), high_freq_range=(8, 15))
        frc_trajectory.append(frc)
        
        # 保存最终预测
        if epoch == epochs - 1:
            predictions_history.append(pred_np)
    
    # 记录结果
    final_frc = frc_trajectory[-1]
    final_test_loss = test_losses[-1]
    avg_frc = np.mean(frc_trajectory[-50:])  # 最后50个epoch的平均FRC
    
    results.append({
        'Regularization': reg_name,
        'Final_FRC': final_frc,
        'Avg_FRC': avg_frc,
        'Final_Test_Loss': final_test_loss,
        'FRC_Trajectory': frc_trajectory,
        'Test_Loss_Trajectory': test_losses
    })

# ==================== 分析 FRC 与泛化的关系 ====================
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

# 1. 绘制 FRC vs Test Loss 散点图
plt.figure(figsize=(10, 6))
for res in results:
    plt.scatter(res['Final_FRC'], res['Final_Test_Loss'], 
                s=100, alpha=0.7, label=res['Regularization'])

plt.xlabel('FRC (Frequency Regularization Coefficient)', fontsize=12)
plt.ylabel('Test Loss', fontsize=12)
plt.title('FRC vs Generalization Performance\n(Higher FRC → Higher Test Loss)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('results/figures/frc_vs_generalization.png', dpi=200, bbox_inches='tight')
plt.close()

# 2. 计算相关性
df_results = pd.DataFrame([{
    'Regularization': r['Regularization'],
    'Final_FRC': r['Final_FRC'],
    'Final_Test_Loss': r['Final_Test_Loss']
} for r in results])

correlation = df_results['Final_FRC'].corr(df_results['Final_Test_Loss'])
print(f"\n{'='*60}")
print(f"FRC 与 Test Loss 的相关性: {correlation:.4f}")
print(f"{'='*60}")

# 3. 绘制 FRC 轨迹
plt.figure(figsize=(12, 6))
for res in results:
    plt.plot(range(1, len(res['FRC_Trajectory']) + 1), 
             res['FRC_Trajectory'], 
             linewidth=2, label=res['Regularization'], alpha=0.8)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('FRC', fontsize=12)
plt.title('FRC Trajectory During Training', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('results/figures/frc_trajectory.png', dpi=200, bbox_inches='tight')
plt.close()

# 4. 保存结果
df_results.to_csv('results/data/exp3_frc_analysis.csv', index=False)

# 5. 绘制 FRC vs Test Loss 的回归线
plt.figure(figsize=(10, 6))
x = df_results['Final_FRC'].values
y = df_results['Final_Test_Loss'].values

plt.scatter(x, y, s=100, alpha=0.7)
# 添加回归线
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", linewidth=2, label=f'Linear fit (corr={correlation:.3f})')

for idx, row in df_results.iterrows():
    plt.annotate(row['Regularization'], 
                (row['Final_FRC'], row['Final_Test_Loss']),
                fontsize=9, alpha=0.7)

plt.xlabel('FRC (Frequency Regularization Coefficient)', fontsize=12)
plt.ylabel('Test Loss', fontsize=12)
plt.title('FRC as Predictor of Generalization\n(FRC ↑ → Test Loss ↑)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/frc_regression.png', dpi=200, bbox_inches='tight')
plt.close()

print("\n✅ Experiment 3 completed!")
print("Results saved to:")
print("  - results/figures/frc_vs_generalization.png")
print("  - results/figures/frc_trajectory.png")
print("  - results/figures/frc_regression.png")
print("  - results/data/exp3_frc_analysis.csv")
print(f"\n关键发现：FRC 与 Test Loss 的相关性 = {correlation:.4f}")
if correlation > 0.5:
    print("✅ 强正相关：FRC 可以作为泛化性能的预测指标！")

