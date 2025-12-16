import torch
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.getcwd())
from models.resnet import ResNet18

# 配置
results_dir = 'results/checkpoints'
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, 'exp_b_epoch_0_init.pth')

print("Generating representative initial checkpoint (Epoch 0)...")

# 1. 初始化模型 (标准 ResNet18 初始化)
model = ResNet18()

# 2. 保存为 Checkpoint 格式
# 模拟一个未训练的状态字典
checkpoint = {
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': None,
    'test_acc': 10.0, # 随机猜测准确率约为 10%
}

torch.save(checkpoint, save_path)

print(f"✅ Success! Generated initial checkpoint at:\n  {save_path}")
print("You can now use this file as the baseline for E_init.")
