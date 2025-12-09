# 快速运行指南

## 方法1：使用快速启动脚本（推荐）

```bash
cd /HSS/ljh/FrequencyRegularization
python run_exp1.py
```

## 方法2：直接运行实验脚本

```bash
cd /HSS/ljh/FrequencyRegularization
python experiments/exp1_toy_baseline.py
```

## 方法3：运行正则化对比实验

```bash
cd /HSS/ljh/FrequencyRegularization
python experiments/exp2_toy_regularization.py
```

## 运行前检查

### 1. 安装依赖
```bash
cd /HSS/ljh/FrequencyRegularization
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install torch numpy matplotlib pandas tqdm scipy
```

### 2. 检查项目结构
确保以下文件存在：
- `models/mlp.py`
- `data/toy_data.py`
- `frequency/fft_utils.py`
- `utils/visualization.py`
- `experiments/exp1_toy_baseline.py`

## 预期输出

运行成功后，会在 `results/` 目录下生成：

### 实验1（Baseline）
- `results/figures/flc_toy_baseline.png` - 频率学习曲线
- `results/figures/fit_snapshots.png` - 训练过程快照
- `results/data/exp1_results.csv` - 数值结果

### 实验2（正则化对比）
- `results/figures/flc_low_regularization.png` - 低频学习曲线对比
- `results/figures/flc_high_regularization.png` - 高频学习曲线对比
- `results/figures/auc_comparison.png` - AUC 对比图
- `results/data/exp2_regularization_comparison.csv` - 对比数据

## 常见问题

### Q: 提示 "ModuleNotFoundError"
**A**: 确保在项目根目录运行，并且已安装所有依赖：
```bash
cd /HSS/ljh/FrequencyRegularization
pip install -r requirements.txt
```

### Q: 提示 "No module named 'models'"
**A**: 确保在项目根目录运行脚本，不要在其他目录运行。

### Q: 运行很慢？
**A**: 这个实验应该很快（几秒到几分钟）。如果很慢，检查是否在使用 GPU（代码会自动检测）。

### Q: 没有看到结果图？
**A**: 检查 `results/figures/` 目录，图片会自动保存，不会弹出窗口显示。

