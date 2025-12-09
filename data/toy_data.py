# data/toy_data.py
"""Generate 1D toy data with controllable frequency components"""

import math
import numpy as np

def generate_toy_data(low_freq=1, high_freq=10, high_amp=0.5, 
                      n_grid=1024, n_train=400, noise=0.0, seed=42):
    """
    生成 1D 合成数据: y = sin(low_freq * x) + high_amp * sin(high_freq * x) + noise
    
    Args:
        low_freq: 低频频率（如 1）
        high_freq: 高频频率（如 10）
        high_amp: 高频幅度
        n_grid: 完整 grid 的点数
        n_train: 训练样本数
        noise: 标签噪声标准差
        seed: 随机种子
    
    Returns:
        x_grid: 完整 grid 的 x 坐标
        y_grid: 完整 grid 的 y 值（ground truth）
        x_train: 训练样本的 x 坐标
        y_train: 训练样本的 y 值（可能含噪声）
    """
    np.random.seed(seed)
    
    # 生成完整 grid
    x_grid = np.linspace(0, 2 * math.pi, n_grid, endpoint=False)
    y_grid = (np.sin(low_freq * x_grid) + 
              high_amp * np.sin(high_freq * x_grid))
    
    # 采样训练点
    train_idx = np.random.choice(n_grid, size=n_train, replace=False)
    x_train = x_grid[train_idx]
    y_train = y_grid[train_idx] + np.random.normal(0, noise, size=n_train)
    
    return x_grid, y_grid, x_train, y_train

