import torch
import time

torch.set_num_threads(1)  # Match your single-threaded C implementation

print("CUDA:", torch.cuda.is_available())
print("Num Threads:", torch.set_num_threads(6)) # Threads set to 6
print("Threads:", torch.get_num_threads())

x = torch.randn(1024, 1024, device="cpu")
y = torch.randn(1024, 1024, device="cpu")

# Warm-up (important)
for _ in range(5):
    torch.matmul(x, y)

start = time.perf_counter()

z = torch.matmul(x, y)

end = time.perf_counter()

print(f"Matmul: {(end - start) * 1000:.3f} ms")

