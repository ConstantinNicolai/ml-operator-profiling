import torch
import torch.nn as nn
import time
import os
import argparse
import math


# Set up argument parser
parser = argparse.ArgumentParser(description="Configuration for the convolutional layer")

# Add command-line arguments
parser.add_argument('--in_channels', type=int, default=64, help='Number of input channels')
parser.add_argument('--out_channels', type=int, default=128, help='Number of output channels')
parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for the convolution')
parser.add_argument('--stride', type=int, default=1, help='Stride for the convolution')
parser.add_argument('--padding', type=int, default=1, help='Padding for the convolution')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the input tensor')
parser.add_argument('--ifmap_size', type=int, default=56, help='Size of the input feature map')
parser.add_argument('--num_layers', type=int, default=500, help='Size of the input feature map')
parser.add_argument('--iterations', type=int, default=120000, help='Size of the input feature map')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
in_channels = args.in_channels
out_channels = args.out_channels
kernel_size = args.kernel_size
stride = args.stride
padding = args.padding
batch_size = args.batch_size
ifmap_size = args.ifmap_size
num_layers = args.num_layers
iterations = args.iterations

finishup = """
bg_pids=$(jobs -p)
for pid in $bg_pids; do
    kill $pid
done
"""

input_size = (batch_size, in_channels, ifmap_size, ifmap_size)

conv_layers = []
for _ in range(num_layers):
    layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding).cuda()
    conv_layers.append(layer)

ifmap = torch.randn(input_size).cuda()




def warmup(operator, ifmap, desired_runtime):
    return 0


