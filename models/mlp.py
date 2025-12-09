# models/mlp.py
"""Simple MLP for toy experiments"""

import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, width=64, depth=2, input_dim=1, output_dim=1, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0 and i < depth - 1:  # 不在最后一层前加 dropout
                layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

