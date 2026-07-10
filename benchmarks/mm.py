import os 
import sys 
import torch

x = torch.randn(16, 32)
y = torch.randn(32, 16)

z = torch.matmul(x, y)

print(z)
