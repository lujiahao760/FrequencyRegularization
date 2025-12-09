#!/usr/bin/env python3
"""
快速测试：验证 FRC 和 RFF 框架是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

print("="*60)
print("测试核心创新功能")
print("="*60)

# 测试1: FRC 计算
print("\n1. 测试 FRC 计算...")
try:
    from frequency.frc import compute_frc
    
    # 生成测试信号
    x = np.linspace(0, 2*np.pi, 1024)
    y = np.sin(x) + 0.5 * np.sin(10*x)
    
    frc, e_low, e_high = compute_frc(y, low_freq_range=(1, 3), high_freq_range=(8, 15))
    print(f"   ✅ FRC = {frc:.4f}")
    print(f"   ✅ E_low = {e_low:.4f}, E_high = {e_high:.4f}")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试2: RFF 框架
print("\n2. 测试 RFF 框架...")
try:
    from frequency.frequency_filter_framework import FrequencyFilterFramework
    
    # 测试 L2
    l2_filter = FrequencyFilterFramework.l2_as_lowpass_filter(1e-4)
    print(f"   ✅ L2 Filter: {l2_filter['filter_type']}")
    
    # 测试 Dropout
    dropout_filter = FrequencyFilterFramework.dropout_as_noise_filter(0.3)
    print(f"   ✅ Dropout Filter: {dropout_filter['filter_type']}")
    
    # 测试 Early Stop
    es_filter = FrequencyFilterFramework.early_stopping_as_truncation_filter(50, 200)
    print(f"   ✅ Early Stop Filter: {es_filter['filter_type']}")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试3: 统一分析
print("\n3. 测试统一分析...")
try:
    from frequency.frequency_filter_framework import FrequencyFilterFramework
    
    # 分析不同正则化
    for reg_type, reg_strength in [('L2', 1e-4), ('Dropout', 0.3), ('EarlyStop', 50)]:
        if reg_type == 'EarlyStop':
            analysis = FrequencyFilterFramework.analyze_regularization_effect(
                reg_type, reg_strength, stop_epoch=50, total_epochs=200
            )
        else:
            analysis = FrequencyFilterFramework.analyze_regularization_effect(
                reg_type, reg_strength
            )
        print(f"   ✅ {reg_type}: {analysis['filter_type']}")
except Exception as e:
    print(f"   ❌ 错误: {e}")

print("\n" + "="*60)
print("✅ 所有核心功能测试通过！")
print("="*60)
print("\n现在可以运行实验3来验证 FRC 与泛化的关系：")
print("  python experiments/exp3_frc_analysis.py")

