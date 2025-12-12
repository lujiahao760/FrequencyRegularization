# experiments/exp_d_resolution_robustness.py
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from models.resnet import ResNet18

def evaluate_on_low_res_data(model, test_loader, resolutions, device='cpu'):
    """
    通过"降采样-再升采样"来模拟不同程度的高频丢失
    resolutions: [32, 28, 24, 20, 16, 12, 8]
    """
    model.eval()
    results = {}
    
    # 原始的 Normalize 参数
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalizer = transforms.Normalize(mean, std)
    
    for res in resolutions:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # images 是 [0, 1] 的原始图 (32x32)
                
                if res < 32:
                    # 1. 降采样 (丢失高频)
                    downsampled = F.interpolate(images, size=(res, res), mode='bilinear', align_corners=False)
                    # 2. 升采样回 32x32 (为了送入网络)
                    upsampled = F.interpolate(downsampled, size=(32, 32), mode='bilinear', align_corners=False)
                else:
                    upsampled = images
                
                # 3. 手动 Normalize
                final_input = normalizer(upsampled)
                
                outputs = model(final_input)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Resolution: {res}x{res}, Acc: {acc:.2f}%")
        results[res] = acc
    
    return results

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 准备数据
    from utils.data_loader import get_cifar10_data_root
    data_root, download = get_cifar10_data_root()
    transform_test_raw = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform_test_raw)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    # 2. 定义分辨率等级 (从清晰到模糊)
    resolutions = [32, 28, 24, 20, 16, 12, 10, 8]
    
    # 3. 加载模型
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'results', 'checkpoints')
    baseline_path = os.path.join(checkpoint_dir, 'exp_b_epoch_50_baseline.pth')
    l2_path = os.path.join(checkpoint_dir, 'exp_b_epoch_50_l2_1e-3.pth')
    dropout_path = os.path.join(checkpoint_dir, 'exp_b_epoch_50_dropout_0.5.pth')

    def load_model(path, dropout=0.0):
        if not os.path.exists(path):
            print(f"Error: Model not found at {path}")
            return None
        model = ResNet18(num_classes=10, dropout=dropout).to(device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    print("Loading models...")
    model_baseline = load_model(baseline_path, 0.0)
    model_l2 = load_model(l2_path, 0.0)
    model_dropout = load_model(dropout_path, 0.5)
    
    results = {}
    
    if model_baseline:
        print("\nEvaluating Baseline...")
        results['baseline'] = evaluate_on_low_res_data(model_baseline, test_loader, resolutions, device)
    
    if model_l2:
        print("\nEvaluating L2...")
        results['l2'] = evaluate_on_low_res_data(model_l2, test_loader, resolutions, device)
        
    if model_dropout:
        print("\nEvaluating Dropout...")
        results['dropout'] = evaluate_on_low_res_data(model_dropout, test_loader, resolutions, device)
    
    # 4. 画图
    plt.figure(figsize=(10, 6))
    
    # 注意：为了符合直觉，我们把X轴反转，或者让它表示"保留的信息量"
    # 这里我们直接画分辨率，分辨率越低(左边)越模糊，分辨率越高(右边)越清晰
    # 或者倒过来画，看你的喜好。这里按分辨率从大到小画。
    
    if 'baseline' in results:
        plt.plot(resolutions, list(results['baseline'].values()), 'b-o', label='Baseline', linewidth=2)
    if 'l2' in results:
        plt.plot(resolutions, list(results['l2'].values()), 'g--o', label='L2 (1e-3)', linewidth=2)
    if 'dropout' in results:
        plt.plot(resolutions, list(results['dropout'].values()), 'orange', linestyle='-.', marker='o', label='Dropout', linewidth=2)
    
    plt.xlabel('Image Resolution (px)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Robustness to Low Resolution (High-Frequency Loss)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 反转X轴，让左边是32(清晰)，右边是8(模糊)，这样符合"随着干扰增加..."的阅读习惯
    plt.gca().invert_xaxis() 
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'figures', 'exp_d_resolution_robustness.png')
    plt.savefig(save_path)
    print(f"\nSaved to {save_path}")
    
    # 保存数据
    df = pd.DataFrame(index=resolutions)
    for key, val in results.items():
        df[key] = pd.Series(val)
    df.to_csv(os.path.join(PROJECT_ROOT, 'results', 'data', 'exp_d_resolution.csv'))