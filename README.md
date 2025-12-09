# Frequency-Regularization Framework
# 频率视角下的正则化机制研究

## 项目概述

本项目研究不同正则化方法（L2、Dropout、Early Stopping）如何通过改变神经网络对不同频率成分的学习速度与顺序来影响泛化性能。

**核心创新点：**
- ⭐ **FRC 指标**：提出 Frequency Regularization Coefficient (FRC) 作为新的频率复杂度指标
- ⭐ **RFF 框架**：提出 Regularization-as-Frequency-Filter (RFF) 统一框架
- 提出 Frequency Learning Curve (FLC) 作为正则化效果的量化指标
- 从频率视角统一解释不同正则化机制
- 验证 FRC 与泛化性能的关系

## 项目结构

```
FrequencyRegularization/
├── data/                    # 数据生成和加载
│   ├── toy_data.py         # 1D 合成数据生成
│   └── image_data.py       # 图像数据（MNIST/CIFAR）
├── models/                 # 模型定义
│   ├── mlp.py             # MLP for toy experiments
│   └── cnn.py             # CNN for image experiments
├── frequency/              # 频率分析工具
│   ├── fft_utils.py       # FFT 和频带分解
│   ├── frc.py             # ⭐ FRC 指标计算（核心创新）
│   └── frequency_filter_framework.py # ⭐ RFF 框架（核心创新）
├── experiments/            # 实验脚本
│   ├── exp1_toy_baseline.py      # 实验1: 验证频率偏置
│   ├── exp2_toy_regularization.py # 实验2: 正则化对比
│   └── exp3_frc_analysis.py      # ⭐ 实验3: FRC 与泛化关系（核心创新）
├── theory/                # 理论文档
│   ├── frc_definition.md         # ⭐ FRC 指标定义
│   └── rff_framework.md          # ⭐ RFF 框架理论
├── utils/                  # 工具函数
│   ├── training.py        # 训练循环
│   ├── metrics.py        # 复杂度指标（sharpness, spectral norm）
│   └── visualization.py   # 绘图工具
├── results/               # 实验结果
│   ├── figures/           # 论文图表
│   └── data/              # 数值结果（CSV）
├── configs/               # 实验配置
│   └── default.yaml       # 默认超参数
└── README.md              # 详细说明

```

## 快速开始

### 1. 环境设置

```bash
cd FrequencyRegularization
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install torch torchvision numpy matplotlib pandas tqdm scipy
```

### 2. 运行第一个实验（验证频率偏置）

**方法1：使用快速启动脚本**
```bash
python run_exp1.py
```

**方法2：直接运行实验脚本**
```bash
cd experiments
python exp1_toy_baseline.py
```

**预期输出：**
- `results/figures/flc_toy_baseline.png` - Frequency Learning Curve（应该看到低频先上升）
- `results/figures/fit_snapshots.png` - 训练过程快照（展示模型如何逐步拟合）
- `results/data/exp1_results.csv` - 数值结果（每 epoch 的 EV 值）

**预期现象：**
- 低频 (k=1) 的 Explained Variance 应该比高频 (k=10) 更快达到 1.0
- 这证明神经网络确实先学低频、后学高频

### 3. 运行正则化对比实验（核心实验）

```bash
cd experiments
python exp2_toy_regularization.py
```

**预期输出：**
- `results/figures/flc_low_regularization.png` - 低频学习曲线对比
- `results/figures/flc_high_regularization.png` - 高频学习曲线对比
- `results/figures/auc_comparison.png` - AUC 对比柱状图
- `results/data/exp2_regularization_comparison.csv` - 定量对比数据

**预期现象：**
- 正则化（L2/Dropout/EarlyStop）应该**延缓高频学习**（高频 EV 上升更慢）
- 但**对低频影响较小**（低频 EV 仍然快速上升）
- 这证明正则化通过"抑制高频学习"来改善泛化

## 核心概念

### Frequency Learning Curve (FLC)

FLC 定义为：对于每个频率带 $f$，在训练 epoch $t$ 上计算该频带上预测与真实的 explained variance：

$$EV_f(t) = 1 - \frac{\|y_f - \hat{y}_{t,f}\|^2}{\|y_f\|^2}$$

### 频率学习速度指标

- $s_f$: epoch 数直到 $EV_f(t)$ 超过阈值（如 0.8）
- AUC: $EV_f(t)$ 曲线下的面积

## 实验计划

1. **实验1**: 验证频率偏置（toy, baseline）
2. **实验2**: 正则化对比（toy, 4种正则化）
3. **实验3**: ⭐ FRC 与泛化关系（核心创新实验）
   - 计算不同正则化配置下的 FRC
   - 验证 FRC 与 Test Loss 的相关性
   - 证明 FRC 作为泛化预测指标的有效性

## 预期结果

- 清晰的 FLC 曲线显示低频先学、高频后学
- 正则化延缓高频学习，改善泛化
- 定量指标（AUC, $s_f$）支持结论
- 复杂度代理（sharpness, spectral norm）提供机制解释

## 时间表（3周计划）

### Week 1: 核心实验（必须完成）
- **Day 1-2**: 运行实验1，验证频率偏置现象
- **Day 3-4**: 运行实验2，完成正则化对比
- **Day 5-7**: 分析结果，撰写 Methods 和 Results 部分

### Week 2: 扩展实验（可选但推荐）
- **Day 8-10**: 实验3（噪声敏感性）
- **Day 11-14**: 实验4（合成图像）或实验5（MNIST）

### Week 3: 完善与写作
- **Day 15-18**: 实验6（机制验证），完善所有图表
- **Day 19-21**: 撰写完整论文（9页），整理代码和结果

**注意**：如果时间紧张，可以只完成实验1-2，这已经足够支撑一篇高质量的课程论文。

## 论文结构（9页）

1. Introduction (1页)
2. Related Work (0.5页)
3. Method: Frequency Learning Curve (1.5页)
4. Experiments: Toy + Regularization (2页)
5. Experiments: Real Data (1.5页)
6. Analysis: Mechanism (1页)
7. Conclusion (0.5页)
8. References + Appendix (1页)

## 常见问题

### Q1: 运行实验1时没有看到明显的频率偏置？
**A**: 检查以下几点：
- 确保训练足够 epoch（至少 100-150）
- 检查模型容量（width=64 应该足够）
- 尝试增加高频幅度（high_amp=0.5 → 0.8）

### Q2: 正则化对比实验没有明显差异？
**A**: 
- 增加正则化强度（L2: 1e-4 → 1e-3）
- 增加 Dropout rate（0.3 → 0.5）
- 确保有足够的训练噪声（noise=0.05）

### Q3: 代码运行出错？
**A**: 
- 检查 Python 版本（需要 3.7+）
- 确保安装了所有依赖：`pip install -r requirements.txt`
- 检查路径是否正确（需要在项目根目录运行）

### Q4: 如何扩展到图像数据？
**A**: 参考 `experiments/exp4_image_synthetic.py`（需要实现）或 `exp5_real_data.py`。核心思路：
- 对图像做 2D FFT
- 按径向频率 bins 分组
- 计算每个 bin 的 explained variance

## 代码结构说明

- `models/mlp.py`: 简单的 MLP 模型
- `data/toy_data.py`: 1D 合成数据生成（sin 函数）
- `frequency/fft_utils.py`: FFT 和频率分解工具
- `utils/visualization.py`: 绘图工具
- `experiments/exp*.py`: 各个实验脚本

## 下一步

1. **立即运行实验1**，验证频率偏置现象
2. **运行实验2**，获得核心结果（正则化对比）
3. **分析结果**，开始撰写论文 Methods 和 Results 部分
4. （可选）扩展实验3-6，丰富论文内容

## 联系方式

如有问题，请查看各实验脚本的注释或检查代码中的 print 语句输出。

