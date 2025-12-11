# data/filtered_data.py
"""
过滤数据集：用于验证正则化对高频/低频的影响
- Low-Pass Dataset: 高斯模糊（滤掉高频）
- High-Pass Dataset: 原图减去模糊图（只剩边缘/高频）
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from scipy import ndimage
import torchvision.transforms.functional as F


class LowPassDataset(Dataset):
    """
    低通数据集：对图像进行高斯模糊，滤掉高频成分
    用于验证：正则化模型在低频数据上表现更好
    """
    def __init__(self, base_dataset, sigma=2.0):
        """
        Args:
            base_dataset: 基础数据集（如 CIFAR-10）
            sigma: 高斯模糊的标准差，越大越模糊（高频越少）
        """
        self.base_dataset = base_dataset
        self.sigma = sigma
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # 转换为 numpy
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = np.array(img)
        
        # 处理维度
        if img_np.ndim == 3:  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
        
        # 对每个通道应用高斯模糊
        blurred = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            blurred[:, :, c] = ndimage.gaussian_filter(img_np[:, :, c], sigma=self.sigma)
        
        # 转换回 (C, H, W)
        if blurred.ndim == 3:
            blurred = np.transpose(blurred, (2, 0, 1))
        
        # 转换回 tensor
        blurred_tensor = torch.from_numpy(blurred).float()
        
        return blurred_tensor, label


class HighPassDataset(Dataset):
    """
    高通数据集：原图减去模糊图，只保留高频成分（边缘）
    用于验证：正则化模型在高频数据上表现更差
    """
    def __init__(self, base_dataset, sigma=2.0):
        """
        Args:
            base_dataset: 基础数据集
            sigma: 高斯模糊的标准差
        """
        self.base_dataset = base_dataset
        self.sigma = sigma
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # 转换为 numpy
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = np.array(img)
        
        # 处理维度
        if img_np.ndim == 3:  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
        
        # 计算模糊图
        blurred = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            blurred[:, :, c] = ndimage.gaussian_filter(img_np[:, :, c], sigma=self.sigma)
        
        # 高通 = 原图 - 模糊图
        high_pass = img_np - blurred
        
        # 归一化到 [0, 1] 范围（可选，根据需求调整）
        high_pass = (high_pass - high_pass.min()) / (high_pass.max() - high_pass.min() + 1e-8)
        
        # 转换回 (C, H, W)
        if high_pass.ndim == 3:
            high_pass = np.transpose(high_pass, (2, 0, 1))
        
        # 转换回 tensor
        high_pass_tensor = torch.from_numpy(high_pass).float()
        
        return high_pass_tensor, label


def get_cifar10_filtered_datasets(data_dir=None, sigma=2.0):
    """
    获取 CIFAR-10 的过滤数据集
    
    Args:
        data_dir: 数据目录，如果为 None 则自动检测现有数据集
        sigma: 高斯模糊的标准差
    
    Returns:
        train_original: 原始训练集
        test_original: 原始测试集
        train_lowpass: 低通训练集
        test_lowpass: 低通测试集
        train_highpass: 高通训练集
        test_highpass: 高通测试集
    """
    # 自动检测现有数据集
    if data_dir is None:
        from utils.data_loader import get_cifar10_data_root
        data_dir, download = get_cifar10_data_root()
    else:
        download = True
    
    # 基础变换
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 原始数据集
    train_original = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform_train
    )
    test_original = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform_test
    )
    
    # 过滤数据集
    train_lowpass = LowPassDataset(train_original, sigma=sigma)
    test_lowpass = LowPassDataset(test_original, sigma=sigma)
    train_highpass = HighPassDataset(train_original, sigma=sigma)
    test_highpass = HighPassDataset(test_original, sigma=sigma)
    
    return {
        'train_original': train_original,
        'test_original': test_original,
        'train_lowpass': train_lowpass,
        'test_lowpass': test_lowpass,
        'train_highpass': train_highpass,
        'test_highpass': test_highpass,
    }


def add_gaussian_noise(dataset, noise_std=0.1):
    """
    给数据集添加高斯噪声（高频噪声）
    用于实验C：高频噪声鲁棒性测试
    
    Args:
        dataset: 基础数据集
        noise_std: 噪声标准差
    
    Returns:
        NoisyDataset: 带噪声的数据集
    """
    class NoisyDataset(Dataset):
        def __init__(self, base_dataset, noise_std):
            self.base_dataset = base_dataset
            self.noise_std = noise_std
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            img, label = self.base_dataset[idx]
            
            if isinstance(img, torch.Tensor):
                noise = torch.randn_like(img) * self.noise_std
                noisy_img = torch.clamp(img + noise, 0, 1)
            else:
                noise = np.random.normal(0, self.noise_std, img.shape)
                noisy_img = np.clip(img + noise, 0, 1)
                noisy_img = torch.from_numpy(noisy_img).float()
            
            return noisy_img, label
    
    return NoisyDataset(dataset, noise_std)
