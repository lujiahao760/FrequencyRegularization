# utils/frequency.py
"""
频率分析工具 - 核心功能模块
包含：径向频谱分析、FFT工具、FRC指标、SSR指标等
"""

import numpy as np
import torch
from scipy import ndimage


def get_radial_spectrum(img_tensor, normalize=True):
    """
    计算图像或 Feature Map 的径向频谱能量（Radial Spectral Power）
    
    输入: img_tensor (C, H, W) 或 (B, C, H, W) 或 (H, W)
    输出: 1D 数组，表示从低频到高频的能量分布
    
    Args:
        img_tensor: 输入张量，可以是单通道或多通道
        normalize: 是否归一化到 [0, 1]
    
    Returns:
        radial_profile: 1D 数组，长度为 min(H, W) // 2
        frequencies: 对应的频率值（径向距离）
    """
    # 转换为 numpy
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.array(img_tensor)
    
    # 处理维度
    if img.ndim == 4:  # (B, C, H, W)
        # 取第一个样本，所有通道平均
        img = img[0].mean(axis=0)
    elif img.ndim == 3:  # (C, H, W)
        # 所有通道平均
        img = img.mean(axis=0)
    elif img.ndim == 2:  # (H, W)
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {img.shape}")
    
    H, W = img.shape
    
    # 1. 2D 傅里叶变换
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift) ** 2  # 能量谱
    
    # 2. 计算径向平均
    # 创建坐标网格
    center_y, center_x = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    
    # 计算每个点到中心的距离
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)
    
    # 计算径向平均
    max_r = min(H, W) // 2
    radial_profile = np.zeros(max_r)
    
    for radius in range(max_r):
        mask = (r == radius)
        if mask.sum() > 0:
            radial_profile[radius] = mag[mask].mean()
    
    # 归一化
    if normalize and radial_profile.sum() > 0:
        radial_profile = radial_profile / radial_profile.sum()
    
    frequencies = np.arange(max_r)
    
    return radial_profile, frequencies


def get_radial_spectrum_torch(img_tensor, normalize=True):
    """
    PyTorch 版本的径向频谱分析
    
    Args:
        img_tensor: torch.Tensor, shape (C, H, W) 或 (B, C, H, W) 或 (H, W)
        normalize: 是否归一化
    
    Returns:
        radial_profile: torch.Tensor, 1D
        frequencies: torch.Tensor, 1D
    """
    # 处理维度
    if img_tensor.ndim == 4:  # (B, C, H, W)
        img = img_tensor[0].mean(dim=0)
    elif img_tensor.ndim == 3:  # (C, H, W)
        img = img_tensor.mean(dim=0)
    elif img_tensor.ndim == 2:  # (H, W)
        img = img_tensor
    else:
        raise ValueError(f"Unsupported tensor shape: {img_tensor.shape}")
    
    H, W = img.shape
    
    # 2D FFT
    fft = torch.fft.fft2(img)
    fft_shift = torch.fft.fftshift(fft)
    mag = torch.abs(fft_shift) ** 2
    
    # 计算径向平均
    center_y, center_x = H // 2, W // 2
    y = torch.arange(H, device=img.device, dtype=torch.float32)
    x = torch.arange(W, device=img.device, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    r = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    r = r.int()
    
    max_r = min(H, W) // 2
    radial_profile = torch.zeros(max_r, device=img.device)
    
    for radius in range(max_r):
        mask = (r == radius)
        if mask.sum() > 0:
            radial_profile[radius] = mag[mask].mean()
    
    if normalize and radial_profile.sum() > 0:
        radial_profile = radial_profile / radial_profile.sum()
    
    frequencies = torch.arange(max_r, device=img.device)
    
    return radial_profile, frequencies


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
        energy = 0.0
        for k in range(k_min, k_max + 1):
            if k < N:
                energy += np.abs(fft[k]) ** 2
                if k > 0 and k < N - k:
                    energy += np.abs(fft[-k]) ** 2
    else:
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
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy().squeeze()
    
    e_low = compute_frequency_energy(y_pred, low_freq_range)
    e_high = compute_frequency_energy(y_pred, high_freq_range)
    
    if e_low < 1e-12:
        frc = float('inf') if e_high > 1e-12 else 0.0
    else:
        frc = e_high / e_low
    
    return frc, e_low, e_high


def compute_ssr(spectrum_before, spectrum_after, high_freq_threshold=0.5):
    """
    计算 Spectral Suppression Ratio (SSR) - 谱抑制比
    
    SSR = (训练前高频能量 - 训练后高频能量) / 训练前高频能量
    
    SSR > 0: 正则化抑制了高频
    SSR < 0: 正则化引入了高频（如 Dropout）
    SSR ≈ 0: 无明显影响
    
    Args:
        spectrum_before: 训练前的径向频谱 (1D array)
        spectrum_after: 训练后的径向频谱 (1D array)
        high_freq_threshold: 高频阈值（相对于最大频率的比例，0.5 表示后一半）
    
    Returns:
        ssr: Spectral Suppression Ratio
        high_freq_energy_before: 训练前高频能量
        high_freq_energy_after: 训练后高频能量
    """
    if isinstance(spectrum_before, torch.Tensor):
        spectrum_before = spectrum_before.cpu().numpy()
    if isinstance(spectrum_after, torch.Tensor):
        spectrum_after = spectrum_after.cpu().numpy()
    
    # 确定高频范围
    max_freq = len(spectrum_before)
    high_freq_start = int(max_freq * high_freq_threshold)
    
    # 计算高频能量
    high_freq_energy_before = np.sum(spectrum_before[high_freq_start:])
    high_freq_energy_after = np.sum(spectrum_after[high_freq_start:])
    
    # 计算 SSR
    if high_freq_energy_before < 1e-12:
        ssr = 0.0
    else:
        ssr = (high_freq_energy_before - high_freq_energy_after) / high_freq_energy_before
    
    return ssr, high_freq_energy_before, high_freq_energy_after


def analyze_weight_spectrum(weight_tensor, normalize=True):
    """
    分析权重张量的频谱
    
    Args:
        weight_tensor: 权重张量，shape (out_features, in_features) 或 (out, in, H, W)
        normalize: 是否归一化
    
    Returns:
        radial_profile: 径向频谱
        frequencies: 频率值
    """
    if weight_tensor.ndim == 2:
        # 线性层权重，直接当作 2D 图像处理
        return get_radial_spectrum(weight_tensor, normalize=normalize)
    elif weight_tensor.ndim == 4:
        # 卷积层权重 (out_channels, in_channels, H, W)
        # 对所有输出通道平均
        weight_2d = weight_tensor.mean(axis=0).mean(axis=0)  # 平均掉通道维度
        return get_radial_spectrum(weight_2d, normalize=normalize)
    else:
        raise ValueError(f"Unsupported weight tensor shape: {weight_tensor.shape}")
