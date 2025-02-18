import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.benchmark as benchmark
from torchvision.models import swin_v2_b
import math

from torch_profiling_utils.torchinfowriter import TorchinfoWriter



# Hyperparameters
N = 32    # Batch size
L = 224   # Input size (L x L)
Q = 2    # Number of passes (iterations)
rundur = 5  # Minimum benchmark runtime in seconds
runnr = 2    # Number of runs

# Model
model = swin_v2_b(weights=None)

model = model.cuda() 
# model.eval()  # Set to training mode

# num_classes = model.classifier[-1].out_features
# print(f"Number of classes: {num_classes}")


# Dummy Data (Random)
inputs = torch.randn(N, 3, L, L, device="cuda")  # Move to GPU
input_data = inputs

torchinfo_writer = TorchinfoWriter(model,
                                    input_data=input_data,
                                    verbose=0)

torchinfo_writer.construct_model_tree()

torchinfo_writer.show_model_tree(attr_list=['Parameters', 'MACs'])



print("####################################################################")

print(model)

exit()

# output = model(inputs)
# output_shape = output.shape
# num_classes = output_shape[1]
targets = torch.randint(0, num_classes, (N,), device="cuda")
# print(targets.shape)
# targets = torch.randint(0, 1000, (N,), device="cuda")
# print(kappa.shape)

model.train()

# Loss and Optimizer for Benchmarking
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def run_training(operator, optimizer, loss_fn, required_iterations, input_tensor, target_tensor):
    for _ in range(required_iterations):
        print("Before forward pass")
        optimizer.zero_grad()  # Reset gradients
        output = operator(input_tensor)  # Forward pass
        print("Before loss computation")
        loss = loss_fn(output, target_tensor)  # Compute loss
        print("Before backward")
        loss.backward()  # Backward pass
        print("Before Optimizer steps")
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

# RUn Warmup
warmup_result = timer.blocked_autorange(callback=None, min_run_time=16)

# Run Benchmark
profile_result = timer.blocked_autorange(callback=None, min_run_time=rundur * runnr)

# Print Benchmark Results
print(profile_result)
print("mean:", profile_result.mean)
print("times:", profile_result.times)
