import torch

# Scalar
#A scalar is a single number and in tensor-speak it's a zero dimension tensor.

scalar = torch.tensor(7)
print(scalar.item()) # scalar.item() returns the value of the scalar tensor
print(scalar.ndim()) # scalar.ndim returns the number of dimensions of the tensor, which is 0 for a scalar


