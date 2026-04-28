import torch
import torch.nn as nn 

x = torch.randn(10,5,10)

ln = nn.LayerNorm(10)

output = ln(x)

print(output)
