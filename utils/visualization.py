# utils/visualization.py
"""Visualization utilities for frequency experiments"""

import matplotlib.pyplot as plt
import numpy as np

def plot_flc(ev_low, ev_high, epochs, save_path=None, title="Frequency Learning Curve"):
    """
    绘制 Frequency Learning Curve
    
    Args:
        ev_low: 低频 explained variance 列表
        ev_high: 高频 explained variance 列表
        epochs: 总 epoch 数
        save_path: 保存路径（如果 None，则显示）
        title: 图标题
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), ev_low, linewidth=2, label='Low freq (k=1)')
    plt.plot(range(1, epochs + 1), ev_high, linewidth=2, label='High freq (k=10)')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Explained Variance', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved FLC plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_fit_snapshots(x_grid, y_grid, x_train, y_train, snapshots, 
                       save_path=None, n_cols=3):
    """
    绘制训练过程中的拟合快照
    
    Args:
        x_grid: 完整 grid 的 x 坐标
        y_grid: ground truth
        x_train: 训练样本 x
        y_train: 训练样本 y
        snapshots: {epoch: prediction_array}
        save_path: 保存路径
        n_cols: 子图列数
    """
    n_snapshots = len(snapshots)
    n_rows = (n_snapshots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_snapshots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (epoch, pred) in enumerate(sorted(snapshots.items())):
        ax = axes[idx]
        ax.plot(x_grid, y_grid, 'b-', linewidth=2, label='Ground truth', alpha=0.7)
        ax.plot(x_grid, pred, 'r--', linewidth=2, label=f'Model (epoch {epoch})')
        ax.scatter(x_train, y_train, s=15, alpha=0.5, c='green', label='Train samples')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'Epoch {epoch}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_snapshots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved fit snapshots to {save_path}")
    else:
        plt.show()
    plt.close()

