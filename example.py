import torch

# Create tensors x and y
# requires_grad=True is necessary to calculate gradients
x = torch.tensor([16.0, 32.0], requires_grad=True)
y = torch.tensor([10.0, 20.0], requires_grad=True)

# 1. Difference: z = x - y
z = x - y

# 2. Square: o = z^2
o = torch.square(z)

# 3. Sum to find Loss (scalar)
loss = torch.sum(o)

# Perform the backward pass
loss.backward()

print(f"Loss: {loss.item()}")
print(f"x.grad: {x.grad}") # Gradient of Loss w.r.t x
print(f"y.grad: {y.grad}") # Gradient of Loss w.r.t y
