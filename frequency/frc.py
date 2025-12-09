# frequency/frc.py
"""
Frequency Regularization Coefficient (FRC)
核心创新：提出新的频率复杂度指标

FRC(θ) = E_high(θ) / E_low(θ)

其中：
- E_high(θ): 模型在高频成分上的能量
- E_low(θ): 模型在低频成分上的能量

FRC 越大，说明模型更倾向于学习高频（可能过拟合）
FRC 越小，说明模型更倾向于学习低频（可能欠拟合）
"""

import numpy as np
import torch

def compute_frequency_energy(signal, k_range):
    """
    计算信号在特定频率范围内的能量
    
    Args:
        signal: 1D numpy array (real signal)
        k_range: tuple (k_min, k_max) 或单个 k 值
    
    Returns:
        energy: 该频率范围内的能量
    """
    N = len(signal)
    fft = np.fft.fft(signal, n=N)
    
    if isinstance(k_range, tuple):
        k_min, k_max = k_range
        # 计算频率范围内的能量
        energy = 0.0
        for k in range(k_min, k_max + 1):
            if k < N:
                energy += np.abs(fft[k]) ** 2
                if k > 0 and k < N - k:
                    energy += np.abs(fft[-k]) ** 2
    else:
        # 单个频率
        k = int(k_range) % N
        energy = np.abs(fft[k]) ** 2
        if k > 0 and k < N - k:
            energy += np.abs(fft[-k]) ** 2
    
    return energy


def compute_frc(y_pred, low_freq_range=(1, 3), high_freq_range=(8, 15)):
    """
    计算 Frequency Regularization Coefficient (FRC)
    
    FRC = E_high / E_low
    
    Args:
        y_pred: 模型预测值 (1D numpy array)
        low_freq_range: 低频范围 (k_min, k_max)
        high_freq_range: 高频范围 (k_min, k_max)
    
    Returns:
        frc: Frequency Regularization Coefficient
        e_low: 低频能量
        e_high: 高频能量
    """
    # 转换为 numpy array
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy().squeeze()
    
    # 计算低频能量
    e_low = compute_frequency_energy(y_pred, low_freq_range)
    
    # 计算高频能量
    e_high = compute_frequency_energy(y_pred, high_freq_range)
    
    # 避免除零
    if e_low < 1e-12:
        frc = float('inf') if e_high > 1e-12 else 0.0
    else:
        frc = e_high / e_low
    
    return frc, e_low, e_high


def compute_frc_trajectory(model_predictions, low_freq_range=(1, 3), high_freq_range=(8, 15)):
    """
    计算训练过程中 FRC 的变化轨迹
    
    Args:
        model_predictions: list of predictions at different epochs
        low_freq_range: 低频范围
        high_freq_range: 高频范围
    
    Returns:
        frc_trajectory: list of FRC values
        e_low_trajectory: list of low-frequency energies
        e_high_trajectory: list of high-frequency energies
    """
    frc_traj = []
    e_low_traj = []
    e_high_traj = []
    
    for pred in model_predictions:
        frc, e_low, e_high = compute_frc(pred, low_freq_range, high_freq_range)
        frc_traj.append(frc)
        e_low_traj.append(e_low)
        e_high_traj.append(e_high)
    
    return frc_traj, e_low_traj, e_high_traj

