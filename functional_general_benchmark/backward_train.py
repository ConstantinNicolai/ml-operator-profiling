import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark

# Define layers outside functions to avoid re-initialization overhead
linear_layer = nn.Linear(200, 300)
relu_activation = nn.ReLU(inplace=False)
relu_inplace_activation = nn.ReLU(inplace=True)

# Define test inputs
input_tensor = torch.randn(10, 200)
gradient = torch.randn(10, 300)

# Functions to benchmark
def lin(input_tensor: torch.Tensor) -> torch.Tensor:
    output = linear_layer(input_tensor)
    return output

def linrelu(input_tensor: torch.Tensor) -> torch.Tensor:
    output = relu_activation(linear_layer(input_tensor))
    return output

def linrelu_inplace(input_tensor: torch.Tensor) -> torch.Tensor:
    output = relu_inplace_activation(linear_layer(input_tensor))
    return output

def lintrain(input_tensor: torch.Tensor, gradient: torch.Tensor) -> None:
    output = linear_layer(input_tensor)
    output.backward(gradient)

def linrelutrain(input_tensor: torch.Tensor, gradient: torch.Tensor) -> None:
    output = relu_activation(linear_layer(input_tensor))
    output.backward(gradient)

def linrelu_inplace_train(input_tensor: torch.Tensor, gradient: torch.Tensor) -> None:
    output = relu_inplace_activation(linear_layer(input_tensor))
    output.backward(gradient)

# Reusable benchmarking function
def benchmark_function(stmt, setup, globals, label, sub_label):
    timer = benchmark.Timer(stmt=stmt, setup=setup, globals=globals)
    profile_result = timer.blocked_autorange(min_run_time=10)
    time_ms = profile_result.mean * 1000  # Convert seconds to milliseconds
    print(f"{label}, {sub_label}: {time_ms:.4f} ms")
    return time_ms

# Benchmarking
time_lin_inference = benchmark_function(
    stmt="lin(input_tensor)",
    setup="from __main__ import lin",
    globals={"input_tensor": input_tensor},
    label="Linear",
    sub_label="Inference",
)

time_linrelu_inference = benchmark_function(
    stmt="linrelu(input_tensor)",
    setup="from __main__ import linrelu",
    globals={"input_tensor": input_tensor},
    label="Linear + ReLU (inplace=False)",
    sub_label="Inference",
)

time_linrelu_inplace_inference = benchmark_function(
    stmt="linrelu_inplace(input_tensor)",
    setup="from __main__ import linrelu_inplace",
    globals={"input_tensor": input_tensor},
    label="Linear + ReLU (inplace=True)",
    sub_label="Inference",
)

time_lin_training = benchmark_function(
    stmt="lintrain(input_tensor, gradient)",
    setup="from __main__ import lintrain",
    globals={"input_tensor": input_tensor, "gradient": gradient},
    label="Linear",
    sub_label="Training",
)

time_linrelu_training = benchmark_function(
    stmt="linrelutrain(input_tensor, gradient)",
    setup="from __main__ import linrelutrain",
    globals={"input_tensor": input_tensor, "gradient": gradient},
    label="Linear + ReLU (inplace=False)",
    sub_label="Training",
)

time_linrelu_inplace_training = benchmark_function(
    stmt="linrelu_inplace_train(input_tensor, gradient)",
    setup="from __main__ import linrelu_inplace_train",
    globals={"input_tensor": input_tensor, "gradient": gradient},
    label="Linear + ReLU (inplace=True)",
    sub_label="Training",
)

# Compute time for ReLU
relu_inference_time = time_linrelu_inference - time_lin_inference
relu_inplace_inference_time = time_linrelu_inplace_inference - time_lin_inference

relu_training_time = time_linrelu_training - time_lin_training
relu_inplace_training_time = time_linrelu_inplace_training - time_lin_training

# Print results for ReLU
print(f"ReLU (inplace=False), Inference: {relu_inference_time:.4f} ms")
print(f"ReLU (inplace=True), Inference: {relu_inplace_inference_time:.4f} ms")
print(f"ReLU (inplace=False), Training: {relu_training_time:.4f} ms")
print(f"ReLU (inplace=True), Training: {relu_inplace_training_time:.4f} ms")
