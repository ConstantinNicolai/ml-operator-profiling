import torch
import torch.nn as nn
import time
import os
import argparse
import math

# Set up argument parser
parser = argparse.ArgumentParser(description="Benchmark PyTorch operations")

# Add common command-line arguments
parser.add_argument('--operation', type=str, required=True, help='Type of PyTorch operation to benchmark (e.g., conv2d, linear, relu, etc.)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the input tensor')
parser.add_argument('--ifmap_size', type=int, default=56, help='Size of the input feature map (only applicable to some operations)')
parser.add_argument('--iterations', type=int, default=120000, help='Number of iterations to run')
parser.add_argument('--in_features', type=int, help='Number of input features (for operations like linear)')
parser.add_argument('--out_features', type=int, help='Number of output features (for operations like linear)')

# Add arguments specific to convolutional layers
parser.add_argument('--in_channels', type=int, help='Number of input channels (for conv operations)')
parser.add_argument('--out_channels', type=int, help='Number of output channels (for conv operations)')
parser.add_argument('--kernel_size', type=int, help='Kernel size for convolution (for conv operations)')
parser.add_argument('--stride', type=int, default=1, help='Stride for the operation (for conv operations)')
parser.add_argument('--padding', type=int, default=0, help='Padding for the operation (for conv operations)')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers (applicable to conv and other operations)')

# Parse the arguments
args = parser.parse_args()

# Map arguments to variables
operation = args.operation
batch_size = args.batch_size
ifmap_size = args.ifmap_size
iterations = args.iterations
num_layers = args.num_layers

# Variables for specific operations
in_channels = args.in_channels
out_channels = args.out_channels
kernel_size = args.kernel_size
stride = args.stride
padding = args.padding
in_features = args.in_features
out_features = args.out_features

# Construct the input tensor
if operation in ['conv2d', 'conv3d']:
    input_size = (batch_size, in_channels, ifmap_size, ifmap_size)
    ifmap = torch.randn(input_size).cuda()
elif operation in ['linear']:
    input_size = (batch_size, in_features)
    ifmap = torch.randn(input_size).cuda()
else:
    raise ValueError(f"Unsupported operation: {operation}")

# Initialize the layers
layers = []
for _ in range(num_layers):
    if operation == 'conv2d':
        layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding).cuda()
    elif operation == 'linear':
        layer = nn.Linear(in_features=in_features, out_features=out_features).cuda()
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    layers.append(layer)

# Warmup
warmup_start_time = time.time()
for i in range(math.ceil(iterations/4)):
    layer = layers[i % num_layers]
    output = layer(ifmap)
warmup_stop_time = time.time()

# Estimate required iterations for 30 seconds benchmark
warmup_time = warmup_stop_time - warmup_start_time
time_per_iteration = warmup_time / math.ceil(iterations/4)
required_iterations = int(30 / time_per_iteration)

# Create the startup command string with parameters
startup = f"""
nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total,pstate --format=csv,noheader,nounits > logs/{operation}_{required_iterations}iter.log &
"""

# Starting the GPU stats logging in the background
os.system(startup)

# Start the timer
start_time = time.time()

# Run the operation in a loop
for i in range(required_iterations):
    layer = layers[i % num_layers]
    output = layer(ifmap)

# Stop the timer
end_time = time.time()

# Stopping the GPU stats logging running in the background
os.system("kill $(jobs -p)")

# Calculate the time taken
total_time = end_time - start_time
print(f"Total time for {required_iterations} iterations: {total_time:.4f} seconds")
