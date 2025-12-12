# experiments/exp_d_blur_robustness.py
import sys
import os
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from models.resnet import ResNet18

def evaluate_on_blurred_data(model, test_loader, kernel_sizes, device='cpu'):
    """
    在不同模糊程度下评估模型
    """
    model.eval()
    results = {}
    
    # 原始的 Normalize 参数
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalizer = transforms.Normalize(mean, std)
    
    for k_size in kernel_sizes:
        correct = 0
        total = 0
        
        # 定义模糊变换 (必须是奇数)
        # sigma 设为 None 让它根据 kernel size 自动计算，或者手动指定
        if k_size == 0:
            blurrer = None
        else:
            blurrer = transforms.GaussianBlur(kernel_size=k_size, sigma=(k_size * 0.3 + 0.8))
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # images 是 [0, 1] 的原始图
                
                if blurrer:
                    blurred_images = blurrer(images)
                else:
                    blurred_images = images
                
                # 手动 Normalize
                final_input = normalizer(blurred_images)
                
                outputs = model(final_input)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Blur Kernel: {k_size}, Acc: {acc:.2f}%")
        results[k_size] = acc
    
    return results

# ... (加载模型的部分和 exp_c 一样，省略以节省篇幅，直接复制下面的 Main) ...

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 准备数据 (无 Normalize)
    from utils.data_loader import get_cifar10_data_root
    data_root, download = get_cifar10_data_root()
    transform_test_raw = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform_test_raw)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    # 2. 定义模糊等级 (奇数)
    # 0 表示不模糊，数值越大越模糊
    blur_levels = [0, 3, 5, 7, 9, 11] 
    
    # 3. 加载模型 (请确保路径正确)
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'results', 'checkpoints')
    # 注意：这里要用你刚才 exp_b 跑出来的真实路径
    baseline_path = os.path.join(checkpoint_dir, 'exp_b_epoch_50_baseline.pth')
    l2_path = os.path.join(checkpoint_dir, 'exp_b_epoch_50_l2_1e-3.pth')
    dropout_path = os.path.join(checkpoint_dir, 'exp_b_epoch_50_dropout_0.5.pth')

    def load_model(path, dropout=0.0):
        model = ResNet18(num_classes=10, dropout=dropout).to(device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    print("Loading models...")
    model_baseline = load_model(baseline_path, 0.0)
    model_l2 = load_model(l2_path, 0.0)
    model_dropout = load_model(dropout_path, 0.5)
    
    results = {}
    print("\nEvaluating Baseline...")
    results['baseline'] = evaluate_on_blurred_data(model_baseline, test_loader, blur_levels, device)
    
    print("\nEvaluating L2...")
    results['l2'] = evaluate_on_blurred_data(model_l2, test_loader, blur_levels, device)
    
    print("\nEvaluating Dropout...")
    results['dropout'] = evaluate_on_blurred_data(model_dropout, test_loader, blur_levels, device)
    
    # 4. 画图
    plt.figure(figsize=(10, 6))
    plt.plot(blur_levels, list(results['baseline'].values()), 'b-o', label='Baseline', linewidth=2)
    plt.plot(blur_levels, list(results['l2'].values()), 'g--o', label='L2 (1e-3)', linewidth=2)
    plt.plot(blur_levels, list(results['dropout'].values()), 'orange', linestyle='-.', marker='o', label='Dropout', linewidth=2)
    
    plt.xlabel('Blur Kernel Size (Low-Pass Filter Strength)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Robustness to High-Frequency Removal (Blur)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'figures', 'exp_d_blur_robustness.png')
    plt.savefig(save_path)
    print(f"\nSaved to {save_path}")