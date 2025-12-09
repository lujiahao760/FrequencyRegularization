# experiments/exp1_toy_baseline.py
"""
实验1: 验证频率偏置（Baseline）
目标：证明神经网络在训练时先学低频、后学高频
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
from utils.visualization import plot_flc, plot_fit_snapshots

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 数据生成 ====================
print("Generating toy data...")
x_grid, y_grid, x_train, y_train = generate_toy_data(
    low_freq=1,      # sin(x)
    high_freq=10,    # sin(10x)
    high_amp=0.5,   # 高频幅度
    n_train=400,    # 训练样本数
    noise=0.0        # 无噪声（baseline）
)

# 转换为 tensor
x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
x_grid_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1).to(device)
y_grid_t = torch.tensor(y_grid, dtype=torch.float32).unsqueeze(1).to(device)

# ==================== 模型 ====================
model = SimpleMLP(width=64, depth=2).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {n_params} parameters")

# ==================== 训练设置 ====================
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
epochs = 150

# 记录指标
ev_low = []   # 低频 explained variance
ev_high = []  # 高频 explained variance
loss_log = []
snapshot_epochs = [1, 10, 50, 100, 150]  # 保存快照的 epoch
snapshots = {}  # {epoch: prediction}

print("Training...")
for epoch in trange(epochs):
    # 训练
    model.train()
    optimizer.zero_grad()
    out = model(x_train_t)
    loss = F.mse_loss(out, y_train_t)
    loss.backward()
    optimizer.step()
    loss_log.append(loss.item())
    
    # 评估（在完整 grid 上）
    model.eval()
    with torch.no_grad():
        pred_grid = model(x_grid_t).cpu()
    
    # 计算频率带上的 explained variance
    ev1 = explained_variance_band(y_grid, pred_grid.numpy().squeeze(), k_target=1)
    ev10 = explained_variance_band(y_grid, pred_grid.numpy().squeeze(), k_target=10)
    ev_low.append(ev1)
    ev_high.append(ev10)
    
    # 保存快照
    if (epoch + 1) in snapshot_epochs:
        snapshots[epoch + 1] = pred_grid.numpy().squeeze()

# ==================== 保存结果 ====================
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

# 1. 绘制 FLC
plot_flc(ev_low, ev_high, epochs, 
         save_path='results/figures/flc_toy_baseline.png',
         title='Frequency Learning Curve (Baseline)')

# 2. 绘制拟合快照
plot_fit_snapshots(x_grid, y_grid, x_train, y_train, snapshots,
                   save_path='results/figures/fit_snapshots.png')

# 3. 保存数值结果
df = pd.DataFrame({
    'epoch': list(range(1, epochs + 1)),
    'loss': loss_log,
    'ev_low_k1': ev_low,
    'ev_high_k10': ev_high
})
df.to_csv('results/data/exp1_results.csv', index=False)

print("\n✅ Experiment 1 completed!")
print("Results saved to:")
print("  - results/figures/flc_toy_baseline.png")
print("  - results/figures/fit_snapshots.png")
print("  - results/data/exp1_results.csv")
print("\n观察：低频 (k=1) 的 EV 应该比高频 (k=10) 更快上升！")

