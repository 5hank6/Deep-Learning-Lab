import torch

# Tensor initialization
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.ones((2, 2))
c = torch.zeros((2, 2))

print("Tensor a:\n", a)
print("Ones tensor:\n", b)
print("Zeros tensor:\n", c)

# Arithmetic operations
print("Addition:\n", a + b)
print("Multiplication:\n", a * b)

# Broadcasting
d = torch.tensor([1, 2])
print("Broadcasting:\n", a + d)

# Indexing & slicing
print("First row:", a[0])
print("Element [1,1]:", a[1, 1])

# Reshaping
e = torch.arange(6)
print("Reshaped tensor:\n", e.reshape(2, 3))

# Autograd
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print("Gradient:", x.grad)
