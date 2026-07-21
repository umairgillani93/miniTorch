import os 
import sys 
import torch
import time


def compare_matmul():
    
    x = torch.randn(1024, 1024)
    y = torch.randn(1024, 1024)

    z = torch.matmul(x, y)
    return z


if __name__ == "__main__":
    start = time.time()
    res = compare_matmul()
    end = time.time()
    print((end - start) * 1000)


