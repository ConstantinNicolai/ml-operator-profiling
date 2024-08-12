import torch
import torch.nn as nn
import time
import os
import argparse

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



# # Configuration for the convolutional layer
# in_channels = 64
# out_channels = 128
# kernel_size = 3
# stride = 1
# padding = 1

# # Assume the input tensor size after initial downsampling is (batch_size, 64, 56, 56)
# batch_size = 32
# ifmap_size = 56
input_size = (batch_size, in_channels, ifmap_size, ifmap_size)

num_layers = 5000 

# Number of iterations to run
iterations = 20000



startup = """
nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits >> logs/conv2d_${in_channels}.log &
"""

# --in_channels "$in_channels" \
#         --out_channels "$out_channels" \
#         --kernel_size "$kernel_size" \
#         --stride "$stride" \
#         --padding "$padding" \
#         --batch_size "$batch_size" \
#         --ifmap_size "$ifmap_size" \


finishup = """
bg_pids=$(jobs -p)
for pid in $bg_pids; do
    kill $pid
done
"""

conv_layers = []
for _ in range(num_layers):
    layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding).cuda()
    conv_layers.append(layer)

ifmap = torch.randn(input_size).cuda()

# Starting the gpu stats logging in the background
os.system(startup)

# Start the timer
start_time = time.time()


# Run the convolution operation in a loop, accessing layers linearly
for i in range(iterations):
    # Linearly access the convolutional layer from the pre-created list
    conv_layer = conv_layers[i % num_layers]
    
    # Apply the convolution operation
    output = conv_layer(ifmap)

# Stop the timer
end_time = time.time()

# Stopping the gpu stats logging running in the background
os.system(finishup)

# Calculate the time taken
total_time = end_time - start_time
print(f"Total time for {iterations} iterations: {total_time:.4f} seconds")


# Cleanup
del ifmap
for layer in conv_layers:
    del layer
torch.cuda.empty_cache()