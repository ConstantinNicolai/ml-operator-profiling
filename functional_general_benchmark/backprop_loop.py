import torch
import torch.nn as nn
import time

# Define a single Conv2d layer
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
input_tensor = torch.randn(1, 1, 5, 5, requires_grad=True)  # Input tensor
target = torch.ones_like(input_tensor)  # Dummy target

# Forward pass (needed to define the computation graph)
output = conv_layer(input_tensor)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

# Warm-up backward pass (to ensure any caching or setup happens)
loss.backward(retain_graph=True)
conv_layer.zero_grad()  # Clear gradients for proper timing

# Time the backward pass for 1000 iterations
start_time = time.time()
for _ in range(4):
    loss.backward(retain_graph=True)  # Use retain_graph to keep the computation graph
    conv_layer.zero_grad()  # Clear gradients for the next iteration
end_time = time.time()

print(f"Time for 4 backward passes: {end_time - start_time:.4f} seconds")
