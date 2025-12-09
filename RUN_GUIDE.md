# 运行实验的完整指南

## 🚀 最简单的运行方式

### 步骤1：进入项目目录
```bash
cd /HSS/ljh/FrequencyRegularization
```

### 步骤2：运行实验1（验证频率偏置）
```bash
python run_exp1.py
```

就这么简单！实验会自动运行，结果会保存在 `results/` 目录下。

---

## 📋 详细步骤说明

### 1. 检查环境
```bash
cd /HSS/ljh/FrequencyRegularization
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### 2. 安装依赖（如果还没安装）
```bash
pip install -r requirements.txt
```

### 3. 运行实验

#### 实验1：验证频率偏置（Baseline）
```bash
python run_exp1.py
# 或
python experiments/exp1_toy_baseline.py
```

**预期时间**：5-10秒  
**输出**：
- `results/figures/flc_toy_baseline.png` - 频率学习曲线
- `results/figures/fit_snapshots.png` - 训练快照
- `results/data/exp1_results.csv` - 数据

#### 实验2：正则化对比（核心实验）
```bash
python experiments/exp2_toy_regularization.py
```

**预期时间**：30-60秒（需要训练4个配置）  
**输出**：
- `results/figures/flc_low_regularization.png`
- `results/figures/flc_high_regularization.png`
- `results/figures/auc_comparison.png`
- `results/data/exp2_regularization_comparison.csv`

---

## 📊 查看结果

### 方法1：直接查看图片
```bash
# 在文件管理器中打开
cd /HSS/ljh/FrequencyRegularization/results/figures
# 然后用图片查看器打开 .png 文件
```

### 方法2：查看数据
```bash
# 查看 CSV 文件
cat results/data/exp1_results.csv | head -20
```

### 方法3：在 Python 中分析
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('results/data/exp1_results.csv')
print(df.head())

# 绘制曲线
plt.plot(df['epoch'], df['ev_low_k1'], label='Low freq')
plt.plot(df['epoch'], df['ev_high_k10'], label='High freq')
plt.legend()
plt.show()
```

---

## ⚠️ 常见问题

### 问题1：`ModuleNotFoundError: No module named 'models'`
**解决**：确保在项目根目录运行
```bash
cd /HSS/ljh/FrequencyRegularization  # 必须在这个目录
python run_exp1.py
```

### 问题2：`FileNotFoundError: results/figures/...`
**解决**：代码会自动创建目录，如果出错，手动创建：
```bash
mkdir -p results/figures results/data
```

### 问题3：运行很慢
**解决**：这个实验应该很快（几秒）。如果很慢：
- 检查是否在 CPU 上运行（正常，不需要 GPU）
- 检查是否有其他进程占用资源

### 问题4：没有看到预期现象
**解决**：
- 检查训练是否完成（应该看到 "✅ Experiment 1 completed!"）
- 查看 `results/data/exp1_results.csv`，检查 `ev_low_k1` 是否比 `ev_high_k10` 更大
- 如果差异不明显，可以增加训练 epoch 数（修改 `epochs = 150` → `epochs = 200`）

---

## 🎯 预期结果

### 实验1应该看到：
1. **FLC 图**：低频曲线（蓝色）应该比高频曲线（红色）更快上升
2. **快照图**：模型先拟合平滑的低频部分，后拟合高频细节
3. **数据**：`ev_low_k1` 的最终值应该接近 1.0，而 `ev_high_k10` 可能较低

### 实验2应该看到：
1. **低频图**：所有正则化方法的低频学习曲线相似
2. **高频图**：正则化方法（L2/Dropout/EarlyStop）的高频学习曲线上升更慢
3. **AUC 图**：正则化方法的高频 AUC 应该更小

---

## 📝 下一步

运行完实验后：
1. 查看生成的图片，理解频率偏置现象
2. 分析 CSV 数据，量化学习速度差异
3. 开始撰写论文的 Methods 和 Results 部分

---

**现在就可以开始运行了！** 🎉

```bash
cd /HSS/ljh/FrequencyRegularization
python run_exp1.py
```

