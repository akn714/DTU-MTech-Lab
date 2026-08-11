"""
# TENSORs
In Python, a tensor is a multi-dimensional array used as the core data structure for machine learning and deep learning frameworks like PyTorch and TensorFlow. It generalizes scalars, vectors, and matrices into any number of dimensions.

1. Scalar (0D Tensor): A single number, e.g., 5.
2. Vector (1D Tensor): A one-dimensional array of numbers, e.g., [1, 2, 3].
3. Matrix (2D Tensor): A two-dimensional array of numbers, e.g., [[1, 2], [3, 4]].
4. Higher-Dimensional Tensors: Tensors can have three or more dimensions, e.g

IMP: Tensors are used to utlize GPU, because while using numpy arrays we can not use GPU, but while using tensors we can use GPU.
"""

import torch

# Creating a 2D Tensor from a Python list
print("==== DEVICE AND TENSOR DETAILS ====")
data = [[1, 2], [3, 4]]
x_tensor = torch.tensor(data)

print(x_tensor)
print(f"Shape: {x_tensor.shape}")  # Outputs torch.Size([2, 2])
print(f"Device: {x_tensor.device}") # Outputs 'cpu' or 'cuda'

print()
print("==== ADDTION, SUBTRACTION AND MULTIPLICATION ON TENSORS ====")
print()

import torch

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y

x = torch.Tensor([1, 2, 3, 4])
y = torch.Tensor([5, 6, 7, 8])

print("Add:", add(x, y))
print("Subtract:", sub(x, y))
print("Multiply:", mul(x, y))