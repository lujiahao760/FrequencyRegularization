import torch
import os
import sys

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

from models.resnet import ResNet18

def generate_init_checkpoint():
    # 路径配置
    save_dir = 'results/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # 这里文件名与您在 calc_new_metric.py 中修改的一致
    save_path = os.path.join(save_dir, 'exp_b_epoch_0_init.pth')

    print(f"Generating initial checkpoint at: {save_path}")
    print("Initializing ResNet18 with standard random weights (Kaiming/Xavier)...")
    
    # 初始化模型
    # 注意：ResNet18 默认就会进行标准的随机初始化
    model = ResNet18(num_classes=10)

    # 构造 checkpoint 字典，模拟真实训练保存的格式
    checkpoint = {
        'epoch': 0,  # 标记为 0，代表未训练状态
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None, # 初始状态没有优化器历史
        'test_acc': 10.0 # 随机猜测准确率约为 10%
    }

    # 保存文件
    torch.save(checkpoint, save_path)
    print("✅ Success! Initial checkpoint saved.")
    print("Now you can run 'calc_new_metric.py' again.")

if __name__ == "__main__":
    generate_init_checkpoint()
