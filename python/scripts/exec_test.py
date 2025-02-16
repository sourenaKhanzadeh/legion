import torch

print("Hello, World!", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

A = torch.randn(10, 10)
B = torch.randn(10, 10)

C = A @ B
print(C)