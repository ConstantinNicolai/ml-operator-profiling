import torch
import torch.nn as nn
import time
import os

# Configuration for the convolutional layer
in_channels = 64
out_channels = 128
kernel_size = 3
stride = 1
padding = 1

# Assume the input tensor size after initial downsampling is (batch_size, 64, 56, 56)
batch_size = 32
input_size = (batch_size, in_channels, 56, 56)

num_layers = 5000 

# Number of iterations to run
iterations = 50000



startup = """
nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits >> logs/gpu_usage_${SLURM_JOB_ID}.log &
"""

finishup = """
bg_pids=$(jobs -p)
for pid in $bg_pids; do
    kill $pid
done
"""

for _ in range(num_layers):
    layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding).cuda()
    conv_layers.append(layer)
