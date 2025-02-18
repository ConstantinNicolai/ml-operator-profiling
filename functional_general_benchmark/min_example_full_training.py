# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.models import convnext_base

# # Hyperparameters
# N = 32    # Batch size
# L = 384  # Input size (L x L)
# Q = 1   # Number of passes (iterations)

# # Model
# model = convnext_base(weights=None)  # No pre-trained weights
# model.train()  # Set to training mode

# # Dummy Data (Random)
# inputs = torch.randn(N, 3, L, L)  # Batch of N RGB images of size LxL
# targets = torch.randint(0, 1000, (N,))  # Random labels for 1000 classes

# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Loop for Q passes
# for _ in range(Q):
#     optimizer.zero_grad()       # Reset gradients
#     outputs = model(inputs)     # Forward pass
#     loss = criterion(outputs, targets)  # Compute loss
#     loss.backward()             # Backward pass
#     optimizer.step()            # Update weights

#     print(f"Loss: {loss.item():.4f}")  # Print loss





import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.benchmark as benchmark
from torchvision.models import convnext_base
import math

# Hyperparameters
N = 32    # Batch size
L = 384   # Input size (L x L)
Q = 2    # Number of passes (iterations)
rundur = 10  # Minimum benchmark runtime in seconds
runnr = 2    # Number of runs

# Model
model = convnext_base(weights=None)
model = model.cuda() 
model.train()  # Set to training mode

# Dummy Data (Random)
inputs = torch.randn(N, 3, L, L, device="cuda")  # Move to GPU
targets = torch.randint(0, 1000, (N,), device="cuda")  # Move to GPU


output = model(inputs)

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
target = torch.randn_like(output)
iterations = 20

for i in range(math.ceil(iterations/4)):       
    # Apply the convolution operation
    optimizer.zero_grad()       # Reset gradients
    output = model(inputs)     # Forward pass
    loss = loss_fn(output, target)  # Compute loss
    loss.backward()             # Backward pass
    optimizer.step()
    torch.cuda.synchronize()    


print("warmup complete")


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Training Function
def run_training(operator, optimizer, loss_fn, required_iterations, input_tensor, target_tensor):
    for _ in range(required_iterations):
        optimizer.zero_grad()  # Reset gradients
        output = operator(input_tensor)  # Forward pass
        loss = loss_fn(output, target_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        torch.cuda.synchronize()  # Ensure proper benchmarking
    return loss.item()

# Benchmark Timer
timer = benchmark.Timer(
    stmt="run_training(operator, optimizer, loss_fn, required_iterations, ifmap, target)",
    setup="from __main__ import run_training",
    globals={
        "operator": model,
        "optimizer": optimizer,
        "loss_fn": criterion,
        "required_iterations": Q,
        "ifmap": inputs,
        "target": targets,
    },
    num_threads=1,
    label="Training Latency Measurement",
    sub_label="torch.utils.benchmark"
)

# Run Benchmark
profile_result = timer.blocked_autorange(callback=None, min_run_time=rundur * runnr)

# Print Benchmark Results
print(profile_result)
print("mean:", profile_result.mean)
print("times:", profile_result.times)
