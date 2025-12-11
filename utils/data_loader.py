# utils/data_loader.py
"""
数据加载工具：自动查找现有的 CIFAR-10 数据集
"""

import os


def find_cifar10_dataset():
    """
    自动查找现有的 CIFAR-10 数据集
    
    Returns:
        data_root: 数据集根目录（包含 cifar-10-batches-py 的目录）
        found: 是否找到数据集
    """
    # 获取项目根目录
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    github_root = os.path.dirname(project_root)
    
    # 可能的数据集位置（按优先级排序）
    possible_locations = [
        # 1. Fine-tune-ViT 目录
        os.path.join(github_root, 'Fine-tune-ViT'),
        # 2. ERC_DoubleDescent 目录
        os.path.join(github_root, 'ERC_DoubleDescent', 'data'),
        # 3. FedPGP_Experiment/DATA 目录（虽然只有 CIFAR-100，但也检查一下）
        os.path.join(github_root, 'FedPGP_Experiment', 'DATA'),
        # 4. 项目目录下的 data 文件夹
        os.path.join(project_root, 'data'),
    ]
    
    # 检查每个位置
    for location in possible_locations:
        cifar10_path = os.path.join(location, 'cifar-10-batches-py')
        if os.path.exists(cifar10_path) and os.path.isdir(cifar10_path):
            print(f"✅ Found CIFAR-10 dataset at: {location}")
            return location, True
    
    # 如果都没找到，返回默认位置
    default_location = os.path.join(project_root, 'data')
    print(f"⚠️  CIFAR-10 dataset not found, will use: {default_location}")
    print("   (will download if needed)")
    return default_location, False


def get_cifar10_data_root():
    """
    获取 CIFAR-10 数据集的根目录
    
    Returns:
        data_root: 数据集根目录
        download: 是否需要下载
    """
    data_root, found = find_cifar10_dataset()
    return data_root, not found  # 如果找到了就不需要下载
