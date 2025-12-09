# theory/frc_definition.md
"""
Frequency Regularization Coefficient (FRC)
核心指标定义文档
"""

# Frequency Regularization Coefficient (FRC)

## 定义

**Frequency Regularization Coefficient (FRC)** 是一个新的频率复杂度指标，用于量化模型对不同频率成分的学习倾向。

### 数学定义

```
FRC(θ) = E_high(θ) / E_low(θ)
```

其中：
- `E_high(θ)`: 模型在高频成分上的能量
- `E_low(θ)`: 模型在低频成分上的能量
- `θ`: 模型参数

### 频率能量计算

对于预测信号 `y_pred`，我们使用 FFT 分解：

```
E_low = Σ_{k ∈ [k_low_min, k_low_max]} |FFT(y_pred)[k]|²
E_high = Σ_{k ∈ [k_high_min, k_high_max]} |FFT(y_pred)[k]|²
```

## 物理解释

### FRC 的含义

- **FRC 大** → 高频能量高 → 模型更倾向于学习高频（可能过拟合）
- **FRC 小** → 低频能量高 → 模型更倾向于学习低频（可能欠拟合）
- **FRC 适中** → 平衡学习 → 泛化性能好

### 与泛化的关系

**假设**：
```
FRC ↑ → 高频学习 ↑ → 过拟合风险 ↑ → Test Loss ↑
FRC ↓ → 高频学习 ↓ → 过拟合风险 ↓ → Test Loss ↓
```

**实验验证**：
- 计算不同正则化配置下的 FRC
- 验证 FRC 与 Test Loss 的正相关关系

## 优势

1. **可量化**：提供数值指标，便于比较
2. **可解释**：直接反映频率学习倾向
3. **可预测**：可以预测模型的泛化性能
4. **统一性**：适用于不同正则化方法

## 应用

1. **正则化效果评估**：比较不同正则化方法的 FRC
2. **超参数调优**：选择使 FRC 适中的超参数
3. **早停策略**：当 FRC 超过阈值时停止训练
4. **模型选择**：选择 FRC 较低的模型

## 与现有指标的关系

- **Sharpness**：衡量损失地形的陡峭程度
- **Spectral Norm**：衡量模型的 Lipschitz 常数
- **FRC**：衡量频率学习倾向（**新指标**）

FRC 提供了频率域的独特视角，补充了现有的复杂度指标。

