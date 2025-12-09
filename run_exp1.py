#!/usr/bin/env python3
"""
一键运行实验1：验证频率偏置
"""

import os
import sys

# 确保在项目根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

print("="*70)
print("🚀 Frequency-Regularization Framework")
print("   实验1: 验证频率偏置（Baseline）")
print("="*70)
print()

# 运行实验
if __name__ == "__main__":
try:
        # 直接执行实验脚本
        exec(open('experiments/exp1_toy_baseline.py').read())
        print("\n" + "="*70)
        print("✅ 实验1完成！")
        print("="*70)
        print("\n结果文件：")
        print("  📊 results/figures/flc_toy_baseline.png")
        print("  📊 results/figures/fit_snapshots.png")
        print("  📄 results/data/exp1_results.csv")
        print("\n请查看这些文件来观察频率偏置现象！")
except Exception as e:
    print(f"\n❌ 运行出错: {e}")
    import traceback
    traceback.print_exc()
        print("\n💡 提示：")
        print("  1. 确保在项目根目录运行：cd /HSS/ljh/FrequencyRegularization")
        print("  2. 确保安装了依赖：pip install -r requirements.txt")
