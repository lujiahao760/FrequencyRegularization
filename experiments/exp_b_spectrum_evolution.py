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
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
# 结果保存路径（项目根目录下的 results/）
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

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


def get_gpu_memory_info():
    """
    通过 nvidia-smi 获取真实的 GPU 内存使用情况
    返回: list of dicts, 每个 dict 包含 {'gpu_id': int, 'total_mem': float, 'used_mem': float, 'free_mem': float}
    """
    import subprocess
    import re
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', 
                                '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                gpu_id = int(parts[0])
                total_mem = float(parts[1])  # MB
                used_mem = float(parts[2])   # MB
                free_mem = total_mem - used_mem
                gpu_info.append({
                    'gpu_id': gpu_id,
                    'total_mem': total_mem / 1024,  # GB
                    'used_mem': used_mem / 1024,    # GB
                    'free_mem': free_mem / 1024      # GB
                })
        return gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not query GPU memory via nvidia-smi: {e}")
        return None


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
            # 生成文件名（使用项目根目录下的 results/）
            checkpoint_dir = os.path.join(RESULTS_DIR, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            if config_name:
                filename = os.path.join(checkpoint_dir, f'exp_b_epoch_{epoch+1}_{config_name}.pth')
            else:
                filename = os.path.join(checkpoint_dir, f'exp_b_epoch_{epoch+1}_l2_{l2_reg}.pth')
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
        
        # 检查频谱数据的形状
        if spectra.ndim == 1:
            # 如果只有一个 epoch 的数据，需要扩展维度
            spectra = spectra.reshape(1, -1)
        
        # 对于小卷积核，频谱可能很短（3-10个点），不需要截断
        # 对于大卷积核，可以截断到前 32 个频率（避免图像太大）
        max_freqs = min(spectra.shape[1], 32) if spectra.shape[1] > 32 else spectra.shape[1]
        spectra = spectra[:, :max_freqs]
        
        # 转置：行是频率，列是 epoch
        im = ax.imshow(spectra.T, aspect='auto', cmap='viridis', 
                      interpolation='nearest', origin='lower')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Frequency Index', fontsize=11)
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
    # 支持通过环境变量或命令行参数选择 GPU
    # 用法: python exp_b_spectrum_evolution.py [gpu_id]
    # 或: CUDA_VISIBLE_DEVICES=1 python exp_b_spectrum_evolution.py
    gpu_id = None
    
    # 1. 优先检查命令行参数
    if len(sys.argv) > 1:
        try:
            gpu_id = int(sys.argv[1])
            print(f"Using GPU {gpu_id} (specified via command line)")
        except ValueError:
            print(f"Warning: Invalid GPU ID '{sys.argv[1]}', will auto-select")
    
    # 2. 如果没有指定，检查环境变量
    if gpu_id is None and 'CUDA_VISIBLE_DEVICES' in os.environ:
        try:
            gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
            print(f"Using GPU {gpu_id} (specified via CUDA_VISIBLE_DEVICES)")
        except ValueError:
            pass
    
    # 3. 如果没有指定，自动选择有足够空闲内存的 GPU
    if gpu_id is None and torch.cuda.is_available():
        print("Auto-selecting GPU based on available memory...")
        
        # 优先使用 nvidia-smi 获取真实的 GPU 内存使用情况
        gpu_mem_info = get_gpu_memory_info()
        
        if gpu_mem_info:
            best_gpu = None
            max_free_mem = 0
            
            for info in gpu_mem_info:
                if info['free_mem'] > max_free_mem and info['free_mem'] > 2.0:  # 至少 2GB 空闲
                    max_free_mem = info['free_mem']
                    best_gpu = info['gpu_id']
            
            if best_gpu is not None:
                gpu_id = best_gpu
                print(f"Selected GPU {gpu_id} ({max_free_mem:.2f} GB free via nvidia-smi)")
            else:
                # 如果所有 GPU 都很忙，默认使用 GPU 1（通常比 GPU 0 更空闲）
                gpu_id = 1 if torch.cuda.device_count() > 1 else 0
                print(f"No GPU with sufficient memory found (need >2GB), defaulting to GPU {gpu_id}")
                if gpu_mem_info:
                    print("GPU memory status:")
                    for info in gpu_mem_info:
                        print(f"  GPU {info['gpu_id']}: {info['free_mem']:.2f} GB free / {info['total_mem']:.2f} GB total")
        else:
            # 回退到 PyTorch 的内存查询（不准确，但总比没有好）
            print("Warning: Using PyTorch memory query (may not reflect other processes)")
            best_gpu = None
            max_free_mem = 0
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                try:
                    torch.cuda.set_device(i)
                    reserved = torch.cuda.memory_reserved(i)
                    free_mem = props.total_memory - reserved
                    
                    if free_mem > max_free_mem and free_mem > 2 * 1024 * 1024 * 1024:  # 至少 2GB 空闲
                        max_free_mem = free_mem
                        best_gpu = i
                except:
                    continue
            
            if best_gpu is not None:
                gpu_id = best_gpu
                print(f"Selected GPU {gpu_id} ({max_free_mem / 1024**3:.2f} GB free via PyTorch)")
            else:
                # 如果所有 GPU 都很忙，默认使用 GPU 1
                gpu_id = 1 if torch.cuda.device_count() > 1 else 0
                print(f"No GPU with sufficient memory found, defaulting to GPU {gpu_id}")
    
    # 设置设备
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        torch.cuda.set_device(device.index)
        torch.cuda.empty_cache()  # 清理 GPU 缓存
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device.index)
        
        # 优先使用 nvidia-smi 获取真实内存信息
        gpu_mem_info = get_gpu_memory_info()
        if gpu_mem_info:
            for info in gpu_mem_info:
                if info['gpu_id'] == device.index:
                    print(f"GPU {device.index} ({props.name}): {info['free_mem']:.2f} GB free / {info['total_mem']:.2f} GB total")
                    print(f"  - Used by other processes: {info['used_mem']:.2f} GB")
                    break
            else:
                # 如果 nvidia-smi 没有返回该 GPU 的信息，使用 PyTorch 的数据
                total_mem = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(device.index) / 1024**3
                reserved = torch.cuda.memory_reserved(device.index) / 1024**3
                free_mem = total_mem - reserved
                print(f"GPU {device.index} ({props.name}): {free_mem:.2f} GB free / {total_mem:.2f} GB total (PyTorch view)")
                print(f"  - Allocated: {allocated:.2f} GB")
                print(f"  - Reserved: {reserved:.2f} GB")
        else:
            # 回退到 PyTorch 的内存查询
            total_mem = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device.index) / 1024**3
            reserved = torch.cuda.memory_reserved(device.index) / 1024**3
            free_mem = total_mem - reserved
            print(f"GPU {device.index} ({props.name}): {free_mem:.2f} GB free / {total_mem:.2f} GB total (PyTorch view, may not reflect other processes)")
            print(f"  - Allocated: {allocated:.2f} GB")
            print(f"  - Reserved: {reserved:.2f} GB")
    
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
    
    # 创建结果目录（项目根目录下的 results/）
    os.makedirs(os.path.join(RESULTS_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'data'), exist_ok=True)
    
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
    
    # 绘制结果（保存到项目根目录下的 results/）
    spectrum_evolution_path = os.path.join(RESULTS_DIR, 'figures', 'exp_b_spectrum_evolution.png')
    ssr_comparison_path = os.path.join(RESULTS_DIR, 'figures', 'exp_b_ssr_comparison.png')
    csv_path = os.path.join(RESULTS_DIR, 'data', 'exp_b_results.csv')
    
    plot_spectrum_evolution(results_dict, save_path=spectrum_evolution_path)
    plot_ssr_comparison(results_dict, save_path=ssr_comparison_path)
    
    # 保存数值结果
    df = pd.DataFrame({
        'epoch': list(range(50)),
        'baseline_acc': results_dict['baseline']['test_accs'],
        'l2_acc': results_dict['l2']['test_accs'],
        'dropout_acc': results_dict['dropout']['test_accs'],
    })
    df.to_csv(csv_path, index=False)
    
    print("\n✅ Experiment B completed!")
    print("Results saved to:")
    print(f"  - {spectrum_evolution_path}")
    print(f"  - {ssr_comparison_path}")
    print(f"  - {csv_path}")
    print("\n观察：")
    print("  1. 强 L2 正则化的模型，其卷积核非常'平滑'（高频能量极低）")
    print("  2. SSR 指标量化了高频抑制的程度")

