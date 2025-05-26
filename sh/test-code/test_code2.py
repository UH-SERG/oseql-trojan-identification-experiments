import torch
from torch.autograd import grad

# Define a simple function
def simple_function(x):
    return 3 * x**2 + 2 * x + 1

# Create a tensor and set requires_grad to True to track computation history
x = torch.tensor([2.0])
x=x.clone().detach().requires_grad_()

# Compute the function value
y = simple_function(x)

# Compute the gradient of y with respect to x using torch.autograd.grad
grad_x = grad(y, x)

# Print the results
print("Input x:", x)
print("Function value y:", y)
print("Gradient of y with respect to x:", grad_x[0])

