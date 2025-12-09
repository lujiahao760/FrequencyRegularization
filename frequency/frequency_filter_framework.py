# frequency/frequency_filter_framework.py
"""
Regularization-as-Frequency-Filter (RFF) Framework
核心创新：将正则化统一理解为频率过滤器

这个框架提出：
- L2 正则化 = Low-pass filter（低通滤波器）
- Dropout = Random noise filter（随机降噪滤波）
- Early Stopping = Truncation filter（截断滤波）
"""

import numpy as np

class FrequencyFilterFramework:
    """
    Regularization-as-Frequency-Filter (RFF) Framework
    
    核心思想：不同正则化方法可以理解为对频率域的不同操作
    """
    
    @staticmethod
    def l2_as_lowpass_filter(weight_decay, frequency_response=True):
        """
        L2 正则化 = Low-pass Filter
        
        机制：
        - L2 惩罚大权重 → 抑制高频权重增长
        - 相当于在频率域做低通滤波
        
        Args:
            weight_decay: L2 正则化强度
            frequency_response: 是否返回频率响应
        
        Returns:
            filter_type: "Low-pass"
            cutoff_freq: 截止频率（与 weight_decay 相关）
        """
        # 简化的频率响应模型
        # 实际中，cutoff frequency 与 weight_decay 成反比
        cutoff_freq = 1.0 / (1.0 + weight_decay * 1000)
        
        if frequency_response:
            return {
                'filter_type': 'Low-pass',
                'cutoff_frequency': cutoff_freq,
                'mechanism': 'Suppresses high-frequency weight growth',
                'effect': 'Preserves low-frequency patterns, attenuates high-frequency noise'
            }
        return cutoff_freq
    
    @staticmethod
    def dropout_as_noise_filter(dropout_rate, frequency_response=True):
        """
        Dropout = Random Noise Filter
        
        机制：
        - Dropout 注入随机噪声 → 扰动网络结构
        - 高频特征对噪声更敏感 → 被抑制
        - 相当于在频率域做随机降噪滤波
        
        Args:
            dropout_rate: Dropout 概率
            frequency_response: 是否返回频率响应
        
        Returns:
            filter_type: "Random noise filter"
            noise_level: 噪声水平
        """
        noise_level = dropout_rate
        
        if frequency_response:
            return {
                'filter_type': 'Random noise filter',
                'noise_level': noise_level,
                'mechanism': 'Injects random noise, high-frequency features are more sensitive',
                'effect': 'Suppresses high-frequency patterns through noise injection'
            }
        return noise_level
    
    @staticmethod
    def early_stopping_as_truncation_filter(stop_epoch, total_epochs, frequency_response=True):
        """
        Early Stopping = Truncation Filter
        
        机制：
        - 提前停止训练 → 截断高频学习过程
        - 相当于在频率域做截断滤波
        
        Args:
            stop_epoch: 停止的 epoch
            total_epochs: 总 epoch 数
            frequency_response: 是否返回频率响应
        
        Returns:
            filter_type: "Truncation filter"
            truncation_ratio: 截断比例
        """
        truncation_ratio = stop_epoch / total_epochs if total_epochs > 0 else 1.0
        
        if frequency_response:
            return {
                'filter_type': 'Truncation filter',
                'truncation_ratio': truncation_ratio,
                'mechanism': 'Stops training before high-frequency learning completes',
                'effect': 'Prevents overfitting by truncating high-frequency convergence'
            }
        return truncation_ratio
    
    @staticmethod
    def analyze_regularization_effect(reg_type, reg_strength, **kwargs):
        """
        统一分析不同正则化方法的频率过滤效果
        
        Args:
            reg_type: 'L2', 'Dropout', 'EarlyStop'
            reg_strength: 正则化强度
            **kwargs: 其他参数
        
        Returns:
            filter_analysis: 频率过滤分析结果
        """
        if reg_type == 'L2':
            return FrequencyFilterFramework.l2_as_lowpass_filter(reg_strength)
        elif reg_type == 'Dropout':
            return FrequencyFilterFramework.dropout_as_noise_filter(reg_strength)
        elif reg_type == 'EarlyStop':
            stop_epoch = kwargs.get('stop_epoch', 100)
            total_epochs = kwargs.get('total_epochs', 200)
            return FrequencyFilterFramework.early_stopping_as_truncation_filter(
                stop_epoch, total_epochs
            )
        else:
            return {'filter_type': 'None', 'effect': 'No filtering'}

