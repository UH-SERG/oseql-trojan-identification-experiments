import torch
from torch.autograd import grad

# Example: calculating activation using classifier_weights[0] * vec
classifier_weights = torch.randn(10)
vec = torch.randn(10)

# Calculate activation
activation = classifier_weights[0] * vec

# Enable requires_grad for the activation tensor
activation.requires_grad_()

# Compute gradients
logit = torch.sum(activation)
grads = grad(logit, activation, create_graph=True)

# Now you can use grads for further computations or backpropagation
print(grads)
