# 项目结构说明

## 📁 完整项目结构

```
FrequencyRegularization/
├── .gitignore              # Git 忽略文件配置
├── README.md               # 项目主文档
├── requirements.txt        # Python 依赖
├── main.py                 # 统一入口脚本
│
├── data/                   # 数据模块
│   ├── __init__.py
│   ├── toy_data.py         # 1D 合成数据生成
│   └── filtered_data.py    # 低通/高通过滤数据集
│
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── mlp.py             # MLP for toy experiments
│   ├── cnn.py             # Simple CNN
│   └── resnet.py          # ResNet-18
│
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── frequency.py       # 核心：频率分析工具（径向频谱、SSR等）
│   └── visualization.py   # 可视化工具
│
├── experiments/           # 实验脚本
│   ├── __init__.py
│   ├── exp_a_synthetic.py          # 实验A：合成数据拟合
│   ├── exp_b_spectrum_evolution.py # 实验B：频谱演变
│   └── exp_c_robustness.py         # 实验C：噪声鲁棒性
│
└── results/                # 实验结果（自动生成）
    ├── figures/           # 图表
    ├── data/              # 数值结果（CSV）
    └── checkpoints/       # 模型检查点
```

## 🎯 核心文件说明

### 1. main.py
统一入口，支持命令行参数运行不同实验：
```bash
python main.py --experiment a --epochs 200 --l2_reg 1e-3
```

### 2. utils/frequency.py
核心频率分析工具：
- `get_radial_spectrum()`: 径向频谱分析
- `compute_ssr()`: Spectral Suppression Ratio 指标
- `analyze_weight_spectrum()`: 权重频谱分析
- `explained_variance_band()`: 频率带上的 explained variance

### 3. data/filtered_data.py
过滤数据集：
- `LowPassDataset`: 低通数据集（高斯模糊）
- `HighPassDataset`: 高通数据集（边缘）
- `add_gaussian_noise()`: 添加高频噪声

### 4. 三个实验
- **exp_a_synthetic.py**: 合成数据拟合，展示 Spectral Bias
- **exp_b_spectrum_evolution.py**: 真实数据频谱演变，分析权重变化
- **exp_c_robustness.py**: 高频噪声鲁棒性测试

## 🚀 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行实验**
   ```bash
   # 实验A
   python main.py --experiment a
   
   # 实验B
   python main.py --experiment b
   
   # 实验C（需要先运行实验B）
   python main.py --experiment c
   ```

## 📝 已删除的文件

以下文件已被删除，功能已整合到新结构中：

- ❌ `frequency/` 目录 → ✅ 整合到 `utils/frequency.py`
- ❌ `experiments/exp1_toy_baseline.py` → ✅ 替换为 `exp_a_synthetic.py`
- ❌ `experiments/exp2_toy_regularization.py` → ✅ 功能整合到新实验
- ❌ `experiments/exp3_frc_analysis.py` → ✅ 功能整合到新实验
- ❌ `run_exp1.py` → ✅ 使用 `main.py` 统一入口
- ❌ `test_core_innovations.py` → ✅ 不再需要
- ❌ 所有旧的 markdown 文档 → ✅ 保留 `README.md`
- ❌ `theory/` 目录 → ✅ 理论说明已整合到 README

## ✨ 项目特点

1. **清晰的模块化结构**：数据、模型、工具、实验分离
2. **统一的入口**：`main.py` 支持命令行参数
3. **完整的文档**：README 包含使用说明和理论背景
4. **可扩展性**：易于添加新实验和功能
