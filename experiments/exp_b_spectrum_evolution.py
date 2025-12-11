# experiments/exp_b_spectrum_evolution.py
"""
实验B：真实数据的频谱演变（进阶）
目标：观察模型在训练过程中，"权重"发生了什么变化
操作：
- 训练 ResNet-18 跑 CIFAR-10
- 设置对照组：组1（无正则化），组2（强 L2），组3（强 Dropout）
- 提取第一层卷积核（Filters）
- 使用 frequency.py 计算卷积核的频谱
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd

from models.resnet import ResNet18
from utils.frequency import analyze_weight_spectrum, compute_ssr


def train_resnet(model, train_loader, test_loader, epochs=50, l2_reg=0.0, 
                 device='cpu', save_checkpoints=True, config_name=''):
    """
    训练 ResNet 并记录权重频谱
    
    Args:
        config_name: 配置名称（用于保存检查点文件名）
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, 
                                weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 记录
    train_losses = []
    test_accs = []
    weight_spectra = []  # 每个 epoch 的第一层卷积核频谱
    checkpoints = [0, 10, 25, 50]  # 保存检查点的 epoch
    
    print(f"Training with L2={l2_reg}, config={config_name}...")
    for epoch in trange(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        scheduler.step()
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        
        # 分析第一层卷积核的频谱
        with torch.no_grad():
            first_conv_weight = model.conv1.weight.data
            spectrum, _ = analyze_weight_spectrum(first_conv_weight, normalize=True)
            weight_spectra.append(spectrum.cpu().numpy() if isinstance(spectrum, torch.Tensor) else spectrum)
        
        # 保存检查点
        if save_checkpoints and (epoch + 1) in checkpoints:
            # 生成文件名
            if config_name:
                filename = f'results/checkpoints/exp_b_epoch_{epoch+1}_{config_name}.pth'
            else:
                filename = f'results/checkpoints/exp_b_epoch_{epoch+1}_l2_{l2_reg}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, filename)
    
    return {
        'train_losses': train_losses,
        'test_accs': test_accs,
        'weight_spectra': weight_spectra,
        'final_spectrum': weight_spectra[-1],
        'initial_spectrum': weight_spectra[0] if weight_spectra else None
    }


def plot_spectrum_evolution(results_dict, save_path=None):
    """
    绘制频谱演变热力图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    configs = [
        ('No Regularization', results_dict['baseline']),
        ('Strong L2 (1e-3)', results_dict['l2']),
        ('Strong Dropout (0.5)', results_dict['dropout'])
    ]
    
    for idx, (title, results) in enumerate(configs):
        ax = axes[idx]
        
        # 转换为热力图数据
        spectra = np.array(results['weight_spectra'])
        # 只取前 32 个频率（避免图像太大）
        spectra = spectra[:, :32]
        
        im = ax.imshow(spectra.T, aspect='auto', cmap='viridis', 
                      interpolation='nearest', origin='lower')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{title}\nFinal Acc: {results["test_accs"][-1]:.2f}%', fontsize=12)
        plt.colorbar(im, ax=ax, label='Spectral Power')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved spectrum evolution to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_ssr_comparison(results_dict, save_path=None):
    """
    绘制 SSR (Spectral Suppression Ratio) 对比
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    configs = [
        ('Baseline', results_dict['baseline']),
        ('L2 (1e-3)', results_dict['l2']),
        ('Dropout (0.5)', results_dict['dropout'])
    ]
    
    ssr_values = []
    labels = []
    
    for label, results in configs:
        if results['initial_spectrum'] is not None and results['final_spectrum'] is not None:
            ssr, _, _ = compute_ssr(
                results['initial_spectrum'], 
                results['final_spectrum'],
                high_freq_threshold=0.5
            )
            ssr_values.append(ssr)
            labels.append(label)
    
    bars = ax.bar(labels, ssr_values, color=['blue', 'green', 'orange'], alpha=0.7)
    ax.set_ylabel('SSR (Spectral Suppression Ratio)', fontsize=12)
    ax.set_title('High-Frequency Suppression by Regularization', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # 添加数值标签
    for bar, ssr in zip(bars, ssr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ssr:.3f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved SSR comparison to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据加载
    print("Loading CIFAR-10...")
    
    # 使用工具函数自动查找数据集
    from utils.data_loader import get_cifar10_data_root
    data_root, download = get_cifar10_data_root()
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=download, 
                                     transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=download, 
                                    transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    # 创建结果目录
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    results_dict = {}
    
    # 实验1：无正则化
    print("\n=== Experiment 1: Baseline (No Regularization) ===")
    model_baseline = ResNet18(num_classes=10, dropout=0.0).to(device)
    results_dict['baseline'] = train_resnet(
        model_baseline, train_loader, test_loader, 
        epochs=50, l2_reg=0.0, device=device, config_name='baseline'
    )
    
    # 实验2：强 L2
    print("\n=== Experiment 2: Strong L2 Regularization ===")
    model_l2 = ResNet18(num_classes=10, dropout=0.0).to(device)
    results_dict['l2'] = train_resnet(
        model_l2, train_loader, test_loader, 
        epochs=50, l2_reg=1e-3, device=device, config_name='l2_1e-3'
    )
    
    # 实验3：强 Dropout
    print("\n=== Experiment 3: Strong Dropout ===")
    model_dropout = ResNet18(num_classes=10, dropout=0.5).to(device)
    results_dict['dropout'] = train_resnet(
        model_dropout, train_loader, test_loader, 
        epochs=50, l2_reg=0.0, device=device, config_name='dropout_0.5'
    )
    
    # 绘制结果
    plot_spectrum_evolution(results_dict, 
                           save_path='results/figures/exp_b_spectrum_evolution.png')
    plot_ssr_comparison(results_dict, 
                       save_path='results/figures/exp_b_ssr_comparison.png')
    
    # 保存数值结果
    df = pd.DataFrame({
        'epoch': list(range(50)),
        'baseline_acc': results_dict['baseline']['test_accs'],
        'l2_acc': results_dict['l2']['test_accs'],
        'dropout_acc': results_dict['dropout']['test_accs'],
    })
    df.to_csv('results/data/exp_b_results.csv', index=False)
    
    print("\n✅ Experiment B completed!")
    print("Results saved to:")
    print("  - results/figures/exp_b_spectrum_evolution.png")
    print("  - results/figures/exp_b_ssr_comparison.png")
    print("  - results/data/exp_b_results.csv")
    print("\n观察：")
    print("  1. 强 L2 正则化的模型，其卷积核非常'平滑'（高频能量极低）")
    print("  2. SSR 指标量化了高频抑制的程度")
