import torch
import torch.nn as nn
import time
import multiprocessing
import os
import subprocess


# Configuration for the convolutional layer
in_channels = 64
out_channels = 128
kernel_size = 3
stride = 1
padding = 1

# Assume the input tensor size after initial downsampling is (batch_size, 64, 56, 56)
batch_size = 32
input_size = (batch_size, in_channels, 56, 56)

# Create a large array of random convolutional layers stored in VRAM
num_layers = 30000  # Large number of layers to simulate a large model
conv_layers = []
for _ in range(num_layers):
    layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding).cuda()
    conv_layers.append(layer)

# Create a large array of random input data on the GPU
data_size = (1000,) + input_size[1:]  # Large dataset to simulate caching scenario
input_data = torch.randn(data_size).cuda()

# Number of iterations to run
iterations = 50000


startup = """
gpu_ids=(${CUDA_VISIBLE_DEVICES//,/ })
for gpu_id in "${gpu_ids[@]}"; do
    nvidia-smi -i ${gpu_id} -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits >> logs/gpu_usage_${SLURM_JOB_ID}.log &
done
"""

finishup = """
bg_pids=$(jobs -p)
for pid in $bg_pids; do
    kill $pid
done
"""

test_shell = """
echo "hello world"
"""

#os.system(startup)

# Execute the shell command using subprocess.run()
#subprocess.run(startup, shell=True, check=True)

# Start the timer
start_time = time.time()

# Run the convolution operation in a loop, accessing layers linearly
for i in range(iterations):
    # Linearly access the convolutional layer from the pre-created list
    conv_layer = conv_layers[i % num_layers]
    
    # Index into the data array, using modulo to loop over if necessary
    index = i % data_size[0]
    x = input_data[index:index+1]
    
    # Apply the convolution operation
    output = conv_layer(x)

# Stop the timer
end_time = time.time()

#os.system(finishup)
#subprocess.run(finishup, shell=True, check=True)

# Calculate the time taken
total_time = end_time - start_time
print(f"Total time for {iterations} iterations: {total_time:.4f} seconds")