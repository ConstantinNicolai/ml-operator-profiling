# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.benchmark as benchmark
# from torchvision.models import swin_v2_b
# import math

# # Hyperparameters
# N = 32    # Batch size
# L = 224   # Input size (L x L)
# Q = 2    # Number of passes (iterations)
# rundur = 5  # Minimum benchmark runtime in seconds
# runnr = 2    # Number of runs

# # Model
# model = swin_v2_b(weights=None)

# model = model.cuda() 
# model.train()  # Set to training mode

# # # Dummy Data (Random)
# inputs = torch.randn(N, 3, L, L, device="cuda")  # Move to GPU
# targets = torch.randint(0, 1000, (N,), device="cuda")  # Move to GPU



# # Loss and Optimizer for Benchmarking
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Function
# def run_training(operator, optimizer, loss_fn, required_iterations, input_tensor, target_tensor):
#     for _ in range(required_iterations):
#         optimizer.zero_grad()  # Reset gradients
#         output = operator(input_tensor)  # Forward pass
#         loss = loss_fn(output, target_tensor)  # Compute loss
#         loss.backward()  # Backward pass
#         optimizer.step()  # Update model parameters
#         torch.cuda.synchronize()  # Ensure proper benchmarking
#     return loss.item()

# # Benchmark Timer
# timer = benchmark.Timer(
#     stmt="run_training(operator, optimizer, loss_fn, required_iterations, ifmap, target)",
#     setup="from __main__ import run_training",
#     globals={
#         "operator": model,
#         "optimizer": optimizer,
#         "loss_fn": criterion,
#         "required_iterations": Q,
#         "ifmap": inputs,
#         "target": targets,
#     },
#     num_threads=1,
#     label="Training Latency Measurement",
#     sub_label="torch.utils.benchmark"
# )

# # RUn Warmup
# warmup_result = timer.blocked_autorange(callback=None, min_run_time=16)

# # Run Benchmark
# profile_result = timer.blocked_autorange(callback=None, min_run_time=rundur * runnr)

# # Print Benchmark Results
# print(profile_result)
# print("mean:", profile_result.mean)
# print("times:", profile_result.times)



# print("next steps")



import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.benchmark as benchmark
from torchvision.models import swin_v2_b
import math

# Hyperparameters
N = 32    # Batch size
L = 224   # Input size (L x L)
Q = 2    # Number of passes (iterations)
rundur = 5  # Minimum benchmark runtime in seconds
runnr = 2    # Number of runs

try:
    # Model
    model = swin_v2_b(weights=None)
    model = model.cuda()
    model.train()  # Set to training mode

    # Dummy Data (Random)
    inputs = torch.randn(N, 3, L, L, device="cuda")  # Move to GPU
    targets = torch.randint(0, 1000, (N,), device="cuda")  # Move to GPU

    # Loss and Optimizer for Benchmarking
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

    # Run Warmup
    warmup_result = timer.blocked_autorange(callback=None, min_run_time=16)

    # Run Benchmark
    profile_result = timer.blocked_autorange(callback=None, min_run_time=rundur * runnr)

    # Print Benchmark Results
    print(profile_result)
    print("mean:", profile_result.mean)
    print("times:", profile_result.times)

except torch.cuda.OutOfMemoryError:
    print("CUDA Out of Memory Error: Model is too large for the GPU.")
    torch.cuda.empty_cache()  # Free up memory

print("next steps")
