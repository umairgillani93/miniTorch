import os 
import sys 
import torch
from torchviz import make_dot

x = torch.randn((5,5), requires_grad = True)
y = 2 * x + 5
loss = y.sum()
loss.backward()

make_dot(loss, params = {'x': x})


