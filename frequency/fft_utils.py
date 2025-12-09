# frequency/fft_utils.py
"""Frequency analysis utilities using FFT"""

import numpy as np

def bandfiltered_signal(signal, k_target, n_total=None):
    """
    保留信号中特定频率成分（k_target）及其共轭
    
    Args:
        signal: 1D numpy array (real)
        k_target: 目标频率索引
        n_total: 总长度（如果 None，使用 signal 的长度）
    
    Returns:
        filtered_signal: 只包含目标频率的信号
    """
    if n_total is None:
        n_total = len(signal)
    
    fft = np.fft.fft(signal, n=n_total)
    mask = np.zeros_like(fft, dtype=bool)
    
    k = int(k_target) % n_total
    mask[k] = True
    if k != 0 and k != n_total - k:
        mask[-k] = True
    
    fft_masked = fft * mask
    sig = np.fft.ifft(fft_masked).real
    
    return sig


def explained_variance_band(y_true, y_pred, k_target):
    """
    计算特定频率带上的 explained variance
    
    EV = 1 - ||y_true_band - y_pred_band||^2 / ||y_true_band||^2
    
    Args:
        y_true: ground truth (1D numpy array)
        y_pred: prediction (1D numpy array)
        k_target: 目标频率索引
    
    Returns:
        ev: explained variance (0-1之间，1表示完美拟合)
    """
    y_true_band = bandfiltered_signal(y_true, k_target)
    y_pred_band = bandfiltered_signal(y_pred, k_target)
    
    num = np.sum((y_true_band - y_pred_band) ** 2)
    den = np.sum(y_true_band ** 2)
    
    if den < 1e-12:
        return 0.0
    
    ev = 1.0 - (num / den)
    return max(0.0, min(1.0, ev))  # 限制在 [0, 1]

