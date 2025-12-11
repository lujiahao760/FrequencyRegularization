# experiments/exp_a_synthetic.py
"""
实验A：合成数据拟合（最直观的原理展示）
目标：复现 Spectral Bias 现象
数据：y = sin(5x) + sin(20x) + sin(50x)（包含低、中、高频）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
import imageio

from models.mlp import SimpleMLP
from utils.frequency import explained_variance_band
from utils.visualization import plot_fit_snapshots


def generate_multi_freq_data(n_grid=1024, n_train=400, noise=0.0, seed=42):
    """
    生成多频率合成数据：y = sin(5x) + sin(20x) + sin(50x)
    """
    np.random.seed(seed)
    
    x_grid = np.linspace(0, 2 * math.pi, n_grid, endpoint=False)
    y_grid = (np.sin(5 * x_grid) + 
              np.sin(20 * x_grid) + 
              np.sin(50 * x_grid))
    
    train_idx = np.random.choice(n_grid, size=n_train, replace=False)
    x_train = x_grid[train_idx]
    y_train = y_grid[train_idx] + np.random.normal(0, noise, size=n_train)
    
    return x_grid, y_grid, x_train, y_train


def train_model(x_train, y_train, x_grid, y_grid, epochs=200, l2_reg=0.0, 
                device='cpu', save_gif=True, seed=42):
    """
    训练模型并记录拟合过程
    
    Args:
        seed: 随机种子，确保可重复性
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 转换为 tensor
    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    x_grid_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1).to(device)
    
    # 模型（稍微增大容量，确保有足够能力学习高频）
    model = SimpleMLP(width=128, depth=3).to(device)
    
    # 优化器（使用较小的学习率，让训练更稳定）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 记录
    ev_low = []   # k=5
    ev_mid = []   # k=20
    ev_high = []  # k=50
    snapshots = {}
    # 根据训练轮数调整快照间隔
    if epochs <= 200:
        snapshot_epochs = list(range(0, epochs, 20)) + [epochs-1]
    else:
        snapshot_epochs = list(range(0, epochs, 50)) + [epochs-1]
    
    print(f"Training with L2={l2_reg}...")
    for epoch in trange(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train_t)
        loss = F.mse_loss(out, y_train_t)
        
        # L2 正则化
        if l2_reg > 0:
            l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_reg * l2_loss
        
        loss.backward()
        optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            pred_grid = model(x_grid_t).cpu().numpy().squeeze()
        
        # 计算各频率的 explained variance
        ev5 = explained_variance_band(y_grid, pred_grid, k_target=5)
        ev20 = explained_variance_band(y_grid, pred_grid, k_target=20)
        ev50 = explained_variance_band(y_grid, pred_grid, k_target=50)
        
        ev_low.append(ev5)
        ev_mid.append(ev20)
        ev_high.append(ev50)
        
        # 保存快照
        if epoch in snapshot_epochs:
            snapshots[epoch] = pred_grid
    
    # 生成 GIF
    if save_gif:
        create_fitting_gif(x_grid, y_grid, x_train, y_train, snapshots, 
                          l2_reg, save_path='results/figures/exp_a_fitting.gif')
    
    return {
        'ev_low': ev_low,
        'ev_mid': ev_mid,
        'ev_high': ev_high,
        'snapshots': snapshots,
        'final_pred': pred_grid
    }


def create_fitting_gif(x_grid, y_grid, x_train, y_train, snapshots, 
                      l2_reg, save_path='results/figures/exp_a_fitting.gif'):
    """创建拟合过程的 GIF 动图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    frames = []
    for epoch in sorted(snapshots.keys()):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x_grid, y_grid, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.8)
        ax.plot(x_grid, snapshots[epoch], 'r--', linewidth=2, 
                label=f'Model (epoch {epoch})', alpha=0.8)
        ax.scatter(x_train, y_train, s=20, alpha=0.4, c='green', 
                  label='Train samples', zorder=3)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Fitting Process (L2={l2_reg}) - Epoch {epoch}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 将图形转换为图像数组
        fig.canvas.draw()
        
        # 使用临时文件方法，更可靠
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # 保存为 PNG 然后读取
        fig.savefig(temp_path, dpi=100, bbox_inches='tight')
        frame = imageio.imread(temp_path)
        os.remove(temp_path)
        
        frames.append(frame)
        plt.close(fig)
    
    imageio.mimsave(save_path, frames, fps=2)
    print(f"Saved GIF to {save_path}")


def create_comparison_gif(x_grid, y_grid, x_train, y_train, 
                         snapshots_baseline, snapshots_l2, l2_reg=1e-4,
                         save_path='results/figures/exp_a_comparison.gif'):
    """创建对比 GIF：同时展示 baseline 和 L2 正则化的拟合过程"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 获取所有 epoch
    all_epochs = sorted(set(list(snapshots_baseline.keys()) + list(snapshots_l2.keys())))
    
    frames = []
    for epoch in all_epochs:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Baseline
        ax = axes[0]
        if epoch in snapshots_baseline:
            ax.plot(x_grid, y_grid, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.8)
            ax.plot(x_grid, snapshots_baseline[epoch], 'r--', linewidth=2, 
                    label=f'Model (epoch {epoch})', alpha=0.8)
            ax.scatter(x_train, y_train, s=15, alpha=0.3, c='green', 
                      label='Train samples', zorder=3)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Baseline (No Regularization) - Epoch {epoch}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # L2 Regularization
        ax = axes[1]
        if epoch in snapshots_l2:
            ax.plot(x_grid, y_grid, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.8)
            ax.plot(x_grid, snapshots_l2[epoch], 'orange', linestyle='--', linewidth=2, 
                    label=f'Model (epoch {epoch})', alpha=0.8)
            ax.scatter(x_train, y_train, s=15, alpha=0.3, c='green', 
                      label='Train samples', zorder=3)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'With L2 Regularization ({l2_reg}) - Epoch {epoch}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Frequency Learning Comparison: Baseline vs L2 Regularization', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存为临时文件
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        fig.savefig(temp_path, dpi=100, bbox_inches='tight')
        frame = imageio.imread(temp_path)
        os.remove(temp_path)
        
        frames.append(frame)
        plt.close(fig)
    
    imageio.mimsave(save_path, frames, fps=2)
    print(f"Saved comparison GIF to {save_path}")


def plot_frequency_decomposition(x_grid, y_grid, pred_baseline, pred_l2, 
                                 save_path='results/figures/exp_a_frequency_decomp.png'):
    """绘制频率分解对比：展示不同频率成分的学习情况"""
    from utils.frequency import bandfiltered_signal
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    frequencies = [
        (5, 'Low Frequency (k=5)'),
        (20, 'Mid Frequency (k=20)'),
        (50, 'High Frequency (k=50)')
    ]
    
    for idx, (k, title) in enumerate(frequencies):
        # 提取该频率成分
        y_true_band = bandfiltered_signal(y_grid, k)
        pred_baseline_band = bandfiltered_signal(pred_baseline, k)
        pred_l2_band = bandfiltered_signal(pred_l2, k)
        
        # Baseline
        ax = axes[idx, 0]
        ax.plot(x_grid, y_true_band, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.8)
        ax.plot(x_grid, pred_baseline_band, 'r--', linewidth=2, label='Baseline Model', alpha=0.8)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Baseline: {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # L2 Regularization
        ax = axes[idx, 1]
        ax.plot(x_grid, y_true_band, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.8)
        ax.plot(x_grid, pred_l2_band, 'orange', linestyle='--', linewidth=2, 
                label='L2 Regularized Model', alpha=0.8)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'L2 Regularization: {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Frequency Decomposition: How Each Frequency Component is Learned', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved frequency decomposition to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_frequency_learning_curves_three(results_baseline, results_l2, results_l2_strong, save_path=None):
    """绘制频率学习曲线对比（三组：baseline, L2 moderate, L2 strong）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = len(results_baseline['ev_low'])
    
    # Baseline
    ax = axes[0]
    ax.plot(range(epochs), results_baseline['ev_low'], 'b-', linewidth=2, 
            label='Low freq (k=5)', alpha=0.8)
    ax.plot(range(epochs), results_baseline['ev_mid'], 'g-', linewidth=2, 
            label='Mid freq (k=20)', alpha=0.8)
    ax.plot(range(epochs), results_baseline['ev_high'], 'r-', linewidth=2, 
            label='High freq (k=50)', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Explained Variance', fontsize=12)
    ax.set_title('Baseline (No Regularization)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # L2 Moderate (1e-3)
    ax = axes[1]
    ax.plot(range(epochs), results_l2['ev_low'], 'b-', linewidth=2, 
            label='Low freq (k=5)', alpha=0.8)
    ax.plot(range(epochs), results_l2['ev_mid'], 'g-', linewidth=2, 
            label='Mid freq (k=20)', alpha=0.8)
    ax.plot(range(epochs), results_l2['ev_high'], 'r-', linewidth=2, 
            label='High freq (k=50)', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Explained Variance', fontsize=12)
    ax.set_title('L2 Regularization (1e-3)\nModerate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # L2 Strong (1e-2)
    ax = axes[2]
    ax.plot(range(epochs), results_l2_strong['ev_low'], 'b-', linewidth=2, 
            label='Low freq (k=5)', alpha=0.8)
    ax.plot(range(epochs), results_l2_strong['ev_mid'], 'g-', linewidth=2, 
            label='Mid freq (k=20)', alpha=0.8)
    ax.plot(range(epochs), results_l2_strong['ev_high'], 'r-', linewidth=2, 
            label='High freq (k=50)', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Explained Variance', fontsize=12)
    ax.set_title('L2 Regularization (1e-2)\nStrong (Overall Underfitting)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.suptitle('Frequency Learning Curves: Effect of L2 Regularization Strength', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 生成数据
    print("Generating multi-frequency data...")
    x_grid, y_grid, x_train, y_train = generate_multi_freq_data(
        n_grid=1024, n_train=400, noise=0.0
    )
    
    # 实验1：无正则化（增加训练轮数，确保充分收敛）
    print("\n=== Experiment 1: Baseline (No Regularization) ===")
    print("Training longer to ensure full convergence...")
    results_baseline = train_model(
        x_train, y_train, x_grid, y_grid, 
        epochs=500, l2_reg=0.0, device=device, save_gif=True, seed=42
    )
    
    # 实验2：L2 正则化（使用适度的正则化，但尝试不同的值）
    print("\n=== Experiment 2: With Moderate L2 Regularization (1e-3) ===")
    results_l2 = train_model(
        x_train, y_train, x_grid, y_grid, 
        epochs=500, l2_reg=1e-3, device=device, save_gif=False, seed=42  # 使用 1e-3
    )
    
    # 实验3：更强的 L2 正则化（用于对比）
    print("\n=== Experiment 3: With Strong L2 Regularization (1e-2) ===")
    results_l2_strong = train_model(
        x_train, y_train, x_grid, y_grid, 
        epochs=500, l2_reg=1e-2, device=device, save_gif=False, seed=42
    )
    
    # 创建对比 GIF（使用适度的正则化）
    print("\nCreating comparison GIF...")
    create_comparison_gif(x_grid, y_grid, x_train, y_train,
                         results_baseline['snapshots'], results_l2['snapshots'],
                         l2_reg=1e-3,
                         save_path='results/figures/exp_a_comparison.gif')
    
    # 绘制频率学习曲线对比（三组对比）
    plot_frequency_learning_curves_three(
        results_baseline, results_l2, results_l2_strong,
        save_path='results/figures/exp_a_frequency_curves.png'
    )
    
    # 绘制频率分解对比
    print("\nCreating frequency decomposition plot...")
    plot_frequency_decomposition(x_grid, y_grid, 
                                results_baseline['final_pred'], 
                                results_l2['final_pred'],
                                save_path='results/figures/exp_a_frequency_decomp.png')
    
    # 打印最终结果（三组对比）
    print("\n" + "="*70)
    print("最终结果对比 (Final Results Comparison):")
    print("="*70)
    print(f"{'Metric':<25} {'Baseline':<15} {'L2 (1e-3)':<15} {'L2 (1e-2)':<15}")
    print("-"*70)
    print(f"{'Low freq (k=5) EV':<25} {results_baseline['ev_low'][-1]:<15.4f} {results_l2['ev_low'][-1]:<15.4f} {results_l2_strong['ev_low'][-1]:<15.4f}")
    print(f"{'Mid freq (k=20) EV':<25} {results_baseline['ev_mid'][-1]:<15.4f} {results_l2['ev_mid'][-1]:<15.4f} {results_l2_strong['ev_mid'][-1]:<15.4f}")
    print(f"{'High freq (k=50) EV':<25} {results_baseline['ev_high'][-1]:<15.4f} {results_l2['ev_high'][-1]:<15.4f} {results_l2_strong['ev_high'][-1]:<15.4f}")
    print("="*70)
    
    # 计算相对抑制程度（使用相对变化百分比）
    def calc_relative_suppression(baseline_ev, reg_ev):
        if baseline_ev < 1e-6:
            return 0.0
        return (baseline_ev - reg_ev) / baseline_ev * 100
    
    print("\n相对抑制程度 (Relative Suppression %):")
    print("-"*70)
    low_suppress = calc_relative_suppression(results_baseline['ev_low'][-1], results_l2['ev_low'][-1])
    mid_suppress = calc_relative_suppression(results_baseline['ev_mid'][-1], results_l2['ev_mid'][-1])
    high_suppress = calc_relative_suppression(results_baseline['ev_high'][-1], results_l2['ev_high'][-1])
    
    print(f"Low freq (k=5):  {low_suppress:.2f}%")
    print(f"Mid freq (k=20): {mid_suppress:.2f}%")
    print(f"High freq (k=50): {high_suppress:.2f}%")
    print("-"*70)
    
    # 判断是否选择性抑制高频
    print("\n分析 (Analysis):")
    print("-"*70)
    if high_suppress > 0 and high_suppress > low_suppress and high_suppress > mid_suppress:
        print("✅ L2 正则化成功选择性抑制了高频学习！")
        print(f"   高频抑制 ({high_suppress:.2f}%) > 低频抑制 ({low_suppress:.2f}%)")
    elif high_suppress < 0:
        print("⚠️  高频 EV 反而提升了！可能的原因：")
        print("   1. Baseline 训练不充分，正则化帮助了优化")
        print("   2. 高频 EV 值太小，波动较大")
        print("   3. 需要更长的训练时间让 baseline 充分收敛")
        print(f"   建议：增加训练轮数，或使用更强的正则化")
    elif high_suppress > 0:
        print("⚠️  L2 正则化抑制了高频，但同时也抑制了低频（可能正则化太强）")
    else:
        print("⚠️  高频抑制不明显，可能需要调整正则化强度或训练参数")
    
    # 对比强正则化的情况
    print("\n强正则化 (L2=1e-2) 的影响：")
    low_suppress_strong = calc_relative_suppression(results_baseline['ev_low'][-1], results_l2_strong['ev_low'][-1])
    high_suppress_strong = calc_relative_suppression(results_baseline['ev_high'][-1], results_l2_strong['ev_high'][-1])
    print(f"  低频抑制: {low_suppress_strong:.2f}%, 高频抑制: {high_suppress_strong:.2f}%")
    print("  → 太强的正则化会导致整体欠拟合，而不是选择性抑制高频")
    
    print("\n✅ Experiment A completed!")
    print("Results saved to:")
    print("  - results/figures/exp_a_fitting.gif (Baseline only)")
    print("  - results/figures/exp_a_comparison.gif (Baseline vs L2)")
    print("  - results/figures/exp_a_frequency_curves.png")
    print("  - results/figures/exp_a_frequency_decomp.png")

