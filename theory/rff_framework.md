# theory/rff_framework.md
"""
Regularization-as-Frequency-Filter (RFF) Framework
核心理论框架文档
"""

# Regularization-as-Frequency-Filter (RFF) Framework

## 核心思想

我们提出一个统一框架，将不同正则化方法理解为对频率域的不同操作：

**正则化 = 频率过滤器**

## 框架定义

### 1. L2 正则化 = Low-pass Filter（低通滤波器）

**机制**：
- L2 惩罚大权重 → 抑制高频权重增长
- 相当于在频率域做低通滤波

**数学表达**：
```
L2(θ) = ||θ||²
→ 抑制高频权重
→ 相当于 H_L2(ω) = 1 / (1 + λ·ω²)
```

**效果**：
- ✅ 保留低频模式（有用信息）
- ⏸️ 衰减高频噪声

### 2. Dropout = Random Noise Filter（随机降噪滤波）

**机制**：
- Dropout 注入随机噪声 → 扰动网络结构
- 高频特征对噪声更敏感 → 被抑制
- 相当于在频率域做随机降噪滤波

**数学表达**：
```
Dropout: x → x · mask, mask ~ Bernoulli(p)
→ 注入噪声
→ 高频特征对噪声敏感
→ 相当于 H_Dropout(ω) = 1 - p·ω²
```

**效果**：
- ✅ 通过噪声抑制高频
- ⏸️ 低频特征对噪声更鲁棒

### 3. Early Stopping = Truncation Filter（截断滤波）

**机制**：
- 提前停止训练 → 截断高频学习过程
- 相当于在频率域做截断滤波

**数学表达**：
```
Early Stop: 在 epoch T_stop 停止
→ 截断高频学习
→ 相当于 H_ES(ω) = 1 if ω < ω_cutoff else 0
```

**效果**：
- ✅ 在学到高频噪声之前停止
- ⏸️ 避免过度拟合高频

## 统一框架

所有正则化方法都可以表示为：

```
H_reg(ω) = Filter_function(ω, λ_reg)
```

其中：
- `ω`: 频率
- `λ_reg`: 正则化强度
- `H_reg(ω)`: 频率响应函数

## 实验验证

通过 FRC (Frequency Regularization Coefficient) 指标验证：

```
FRC = E_high / E_low
```

**假设**：
- FRC 越高 → 高频能量越高 → 过拟合风险越大
- 正则化 → 降低 FRC → 改善泛化

## 理论贡献

1. **统一视角**：将不同正则化方法统一理解为频率过滤
2. **可量化指标**：FRC 作为正则化效果的量化工具
3. **预测能力**：FRC 可以预测模型的泛化性能

