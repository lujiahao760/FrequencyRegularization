# main.py
"""
统一入口：支持命令行参数运行不同实验
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_experiment_a(args):
    """运行实验A：合成数据拟合"""
    from experiments.exp_a_synthetic import (
        generate_multi_freq_data, train_model, plot_frequency_learning_curves
    )
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment A: Synthetic Data Fitting")
    print(f"Using device: {device}")
    
    # 生成数据
    x_grid, y_grid, x_train, y_train = generate_multi_freq_data(
        n_grid=1024, n_train=400, noise=0.0
    )
    
    # Baseline
    results_baseline = train_model(
        x_train, y_train, x_grid, y_grid, 
        epochs=args.epochs, l2_reg=0.0, device=device, save_gif=True
    )
    
    # L2 Regularization
    results_l2 = train_model(
        x_train, y_train, x_grid, y_grid, 
        epochs=args.epochs, l2_reg=args.l2_reg, device=device, save_gif=False
    )
    
    # 绘制对比
    plot_frequency_learning_curves(
        results_baseline, results_l2, 
        save_path='results/figures/exp_a_frequency_curves.png'
    )
    
    print("✅ Experiment A completed!")


def run_experiment_b(args):
    """运行实验B：真实数据的频谱演变"""
    from experiments.exp_b_spectrum_evolution import (
        train_resnet, plot_spectrum_evolution, plot_ssr_comparison
    )
    from models.resnet import ResNet18
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch
    import pandas as pd
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment B: Spectrum Evolution")
    print(f"Using device: {device}")
    
    # 数据加载
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
    
    os.makedirs('results/checkpoints', exist_ok=True)
    results_dict = {}
    
    # 三个配置
    configs = [
        ('baseline', 0.0, 0.0),
        ('l2', args.l2_reg, 0.0),
        ('dropout', 0.0, args.dropout)
    ]
    
    for name, l2_reg, dropout in configs:
        print(f"\nTraining {name}...")
        model = ResNet18(num_classes=10, dropout=dropout).to(device)
        results_dict[name] = train_resnet(
            model, train_loader, test_loader, 
            epochs=args.epochs, l2_reg=l2_reg, device=device, config_name=name
        )
    
    # 绘制结果
    plot_spectrum_evolution(results_dict, 
                           save_path='results/figures/exp_b_spectrum_evolution.png')
    plot_ssr_comparison(results_dict, 
                       save_path='results/figures/exp_b_ssr_comparison.png')
    
    # 保存结果
    df = pd.DataFrame({
        'epoch': list(range(args.epochs)),
        'baseline_acc': results_dict['baseline']['test_accs'],
        'l2_acc': results_dict['l2']['test_accs'],
        'dropout_acc': results_dict['dropout']['test_accs'],
    })
    df.to_csv('results/data/exp_b_results.csv', index=False)
    
    print("✅ Experiment B completed!")


def run_experiment_c(args):
    """运行实验C：高频噪声鲁棒性"""
    from experiments.exp_c_robustness import (
        evaluate_on_noisy_data, plot_robustness_curves
    )
    from models.resnet import ResNet18
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch
    import pandas as pd
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment C: Noise Robustness")
    print(f"Using device: {device}")
    
    # 数据加载
    # 使用工具函数自动查找数据集
    from utils.data_loader import get_cifar10_data_root
    data_root, download = get_cifar10_data_root()
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=download, 
                                   transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    # 噪声水平
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    # 加载或训练模型
    checkpoint_dir = 'results/checkpoints'
    models = {}
    
    # 尝试加载，如果不存在则快速训练
    baseline_path = os.path.join(checkpoint_dir, 'exp_b_epoch_50_baseline.pth')
    l2_path = os.path.join(checkpoint_dir, f'exp_b_epoch_50_l2_1e-3.pth')
    dropout_path = os.path.join(checkpoint_dir, 'exp_b_epoch_50_dropout_0.5.pth')
    
    if os.path.exists(baseline_path):
        from experiments.exp_c_robustness import load_trained_model
        models['baseline'] = load_trained_model(baseline_path, dropout=0.0, device=device)
        models['l2'] = load_trained_model(l2_path, dropout=0.0, device=device)
        if os.path.exists(dropout_path):
            models['dropout'] = load_trained_model(dropout_path, dropout=0.5, device=device)
        else:
            models['dropout'] = ResNet18(num_classes=10, dropout=args.dropout).to(device)
    else:
        print("⚠️  Models not found. Please run Experiment B first.")
        print("Training quick models for demonstration...")
        # 这里可以添加快速训练逻辑
        return
    
    # 评估
    results_dict = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results_dict[name] = evaluate_on_noisy_data(
            model, test_loader, noise_levels, device=device
        )
    
    # 绘制结果
    plot_robustness_curves(results_dict, 
                          save_path='results/figures/exp_c_robustness.png')
    
    # 保存结果
    df = pd.DataFrame({
        'noise_std': noise_levels,
        'baseline_acc': [results_dict['baseline'][n] for n in noise_levels],
        'l2_acc': [results_dict['l2'][n] for n in noise_levels],
        'dropout_acc': [results_dict['dropout'][n] for n in noise_levels],
    })
    df.to_csv('results/data/exp_c_robustness.csv', index=False)
    
    print("✅ Experiment C completed!")


def main():
    parser = argparse.ArgumentParser(description='Frequency Regularization Experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['a', 'b', 'c', 'all'],
                       help='Experiment to run: a (synthetic), b (spectrum), c (robustness), all')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--l2_reg', type=float, default=1e-3,
                       help='L2 regularization strength (default: 1e-3)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    parser.add_argument('--analyze_spectrum', action='store_true',
                       help='Analyze spectrum (for experiment b)')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/checkpoints', exist_ok=True)
    
    if args.experiment == 'a':
        run_experiment_a(args)
    elif args.experiment == 'b':
        run_experiment_b(args)
    elif args.experiment == 'c':
        run_experiment_c(args)
    elif args.experiment == 'all':
        print("Running all experiments...")
        run_experiment_a(args)
        run_experiment_b(args)
        run_experiment_c(args)
        print("\n✅ All experiments completed!")


if __name__ == '__main__':
    main()
