import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark


# input_tensor = torch.randn(1, 3, requires_grad = True)
# input_tensor1 = torch.randn(1, 3)

# layer = nn.Linear(in_features=3, out_features=2)
# layer = nn.ReLU(inplace=True)

# output1 = layer(input_tensor1)
# gradient = torch.randn_like(output1)

# output = layer(input_tensor)

# output.backward(gradient)

# input_tensor.grad = None

# for i in range(5):

#     output = layer(input_tensor)

#     output.backward(gradient)

#     input_tensor.grad = None


# # Define the kernel for benchmarking
# def run_training(operators, num_layers: int, required_iterations: int, input_tensor: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
#     #input_tensor.requires_grad = True  # Enable gradient computation
#     input_tensor = input_tensor + 0

#     for k in range(required_iterations):
#         # Linearly access the convolutional layer from the pre-created list
#         operator = operators[k % num_layers].train()

#         # Apply the convolution operation
#         output = operator(input_tensor)

#         output.backward(gradient)

#         # Reset input tensor gradients after each backward pass
#         input_tensor.grad = None


#         # Zero gradients collectively when cycling back to the start of the operator list
#         if (k + 1) % num_layers == 0:
#             with torch.no_grad():
#                 # Collectively zero out all gradients for all operators
#                 for param in [p for op in operators for p in op.parameters() if p.grad is not None]:
#                     param.grad.zero_()

#     return output

input_tensor1 = torch.randn(10, 200)

linear_layer1 = nn.Linear(200, 300)  # Input features: 20, Output features: 30
relu_activation1 = nn.ReLU()

output1 = relu_activation1(linear_layer1(input_tensor1))

gradient = torch.randn_like(output1)



input_tensor = torch.randn(10, 200)

def linrelu(input_tensor: torch.Tensor) -> torch.Tensor:

    linear_layer = nn.Linear(200, 300)  # Input features: 20, Output features: 30
    relu_activation = nn.ReLU()

    output = relu_activation(linear_layer(input_tensor))

    return output


# PyTorch Benchmark Timer
num_repeats = 1  # Number of times to repeat the measurement
timer = benchmark.Timer(
    stmt="linrelu(input_tensor)",  # Statement to benchmark  # Setup the function and variables
    setup="from __main__ import linrelu",
    globals={
        "input_tensor": input_tensor
    },
    num_threads=1,  # Number of threads to use
    label="Latency Measurement",
    sub_label="torch.utils.benchmark"
)


profile_result_train = timer.blocked_autorange(callback=None, min_run_time=10)


time = profile_result_train.mean

print("lin, relu, inference",10000*time)


def linrelutrain(input_tensor: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:

    linear_layer = nn.Linear(200, 300)  # Input features: 20, Output features: 30
    relu_activation = nn.ReLU()

    output = relu_activation(linear_layer(input_tensor))

    res = output.backward(gradient)

    return res


# PyTorch Benchmark Timer
num_repeats = 1  # Number of times to repeat the measurement
timer = benchmark.Timer(
    stmt="linrelutrain(input_tensor, gradient)",  # Statement to benchmark  # Setup the function and variables
    setup="from __main__ import linrelutrain",
    globals={
        "input_tensor": input_tensor,
        "gradient" : gradient
    },
    num_threads=1,  # Number of threads to use
    label="Latency Measurement",
    sub_label="torch.utils.benchmark"
)


profile_result_train = timer.blocked_autorange(callback=None, min_run_time=10)


time = profile_result_train.mean

print("lin, relu, training",10000*time)














def lin(input_tensor: torch.Tensor) -> torch.Tensor:

    linear_layer = nn.Linear(200, 300)  # Input features: 20, Output features: 30

    output = linear_layer(input_tensor)

    return output


# PyTorch Benchmark Timer
num_repeats = 1  # Number of times to repeat the measurement
timer = benchmark.Timer(
    stmt="lin(input_tensor)",  # Statement to benchmark  # Setup the function and variables
    setup="from __main__ import lin",
    globals={
        "input_tensor": input_tensor
    },
    num_threads=1,  # Number of threads to use
    label="Latency Measurement",
    sub_label="torch.utils.benchmark"
)


profile_result_train = timer.blocked_autorange(callback=None, min_run_time=10)


time = profile_result_train.mean

print("lin, inference",10000*time)


def lintrain(input_tensor: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:

    linear_layer = nn.Linear(200, 300)  # Input features: 20, Output features: 30

    output = linear_layer(input_tensor)

    res = output.backward(gradient)

    return res


# PyTorch Benchmark Timer
num_repeats = 1  # Number of times to repeat the measurement
timer = benchmark.Timer(
    stmt="lintrain(input_tensor, gradient)",  # Statement to benchmark  # Setup the function and variables
    setup="from __main__ import lintrain",
    globals={
        "input_tensor": input_tensor,
        "gradient" : gradient
    },
    num_threads=1,  # Number of threads to use
    label="Latency Measurement",
    sub_label="torch.utils.benchmark"
)


profile_result_train = timer.blocked_autorange(callback=None, min_run_time=10)


time = profile_result_train.mean

print("lin, training",10000*time)


















# # Define a ReLU operator
# operator = nn.ReLU(inplace = True)
# layer = nn.Linear(in_features=5, out_features=3)

# # Input tensor
# input_tensor = torch.randn(2, 5)
# input_relu = layer(input_tensor)

# for i in range(30):

#     # Apply ReLU (out-of-place operation)
#     output = operator(input_tensor)  # Out-of-place operation

#     # Backward pass with custom gradient
#     # output.retain_grad()
#     output.backward(gradient)

#     input_tensor.grad = None
