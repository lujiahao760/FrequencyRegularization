import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

from models.resnet import ResNet18
from utils.frequency import analyze_weight_spectrum

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = 'results/checkpoints'
figures_dir = 'results/figures'
os.makedirs(figures_dir, exist_ok=True)

# 模型列表
models_config = [
    ('Baseline', 'exp_b_epoch_50_baseline.pth'),
    ('L2 (1e-3)', 'exp_b_epoch_50_l2_1e-3.pth'),
    ('Dropout (0.5)', 'exp_b_epoch_50_dropout_0.5.pth')
]

# 初始模型 (Epoch 0)
# 我们刚刚生成的代表性初始权重
init_ckpt_path = os.path.join(results_dir, 'exp_b_epoch_0_init.pth') 
# 如果你只保存了 epoch 1, 就用 epoch 1 当作 init，影响不大。
# 如果完全没保存初始权重，可以重新初始化一个 ResNet18 计算 init spectrum。

print("Calculating High-Frequency Growth Rate (HFGR)...")

# 1. 计算初始的高频能量 E_high_init
def get_high_freq_energy(model_state_dict, threshold=0.5):
    # 提取第一层卷积核
    if 'conv1.weight' in model_state_dict:
        weight = model_state_dict['conv1.weight']
    else:
        # Try to find the first conv layer
        for k, v in model_state_dict.items():
            if 'conv' in k and 'weight' in k:
                weight = v
                break
        else:
            raise ValueError("Could not find conv1.weight or similar in state_dict")

    # 使用 your analyze_weight_spectrum (未归一化版本!)
    # Note: analyze_weight_spectrum returns (spectrum, spectrum_2d)
    spectrum, _ = analyze_weight_spectrum(weight, normalize=False)
    
    # 转换为 numpy
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.cpu().numpy()
        
    # 计算高频部分能量 (后 50%)
    cutoff = int(len(spectrum) * threshold)
    if cutoff >= len(spectrum): cutoff = len(spectrum) - 1
    e_high = np.sum(spectrum[cutoff:])
    return e_high

# 加载初始权重
dummy_model = ResNet18()
if os.path.exists(init_ckpt_path):
    print(f"Loading init checkpoint from {init_ckpt_path}")
    ckpt = torch.load(init_ckpt_path, map_location=device)
    E_init = get_high_freq_energy(ckpt['model_state_dict'])
else:
    print(f"Warning: Init checkpoint not found at {init_ckpt_path}, using random init.")
    E_init = get_high_freq_energy(dummy_model.state_dict())

print(f"Initial High-Freq Energy: {E_init:.4f}")

# 2. 计算各模型的 Final E_high 并计算 HFGR
names = []
hfgr_values = []

for name, path in models_config:
    full_path = os.path.join(results_dir, path)
    if not os.path.exists(full_path):
        print(f"Missing: {full_path}")
        # For demonstration purposes if files are missing, we might want to skip or mock
        # But for now, let's continue to see which ones are missing
        continue
        
    ckpt = torch.load(full_path, map_location=device)
    E_final = get_high_freq_energy(ckpt['model_state_dict'])
    
    # 公式：(Final - Init) / Init
    growth_rate = (E_final - E_init) / E_init
    
    names.append(name)
    hfgr_values.append(growth_rate)
    print(f"{name}: E_final={E_final:.4f}, Growth Rate={growth_rate:.4f} ({growth_rate*100:.1f}%)")

if not names:
    print("No models found to plot.")
    sys.exit(1)

# 3. 重新画图 (柱状图)
plt.figure(figsize=(8, 5))
bars = plt.bar(names, hfgr_values, color=['blue', 'green', 'orange'], alpha=0.7)
plt.ylabel('HF Growth Rate ( (E_final - E_init) / E_init )', fontsize=12)
plt.title('High-Frequency Energy Growth by Regularization', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

# 在柱子上标数值
for bar, val in zip(bars, hfgr_values):
    plt.text(bar.get_x() + bar.get_width()/2, val, f'+{val*100:.1f}%', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
save_path = os.path.join(figures_dir, 'fig_ssr_new.png')
plt.savefig(save_path, dpi=300)
print(f"New figure saved to {save_path}")
