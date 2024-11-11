import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.init as init
import pickle
import lzma
import yaml
import os
import numpy as np
import torch.nn.init as init
import time
import math
from datetime import datetime
import uncertainties
import torch.utils.benchmark as benchmark
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new, process_log_file,get_latest_dataset_file, load_latest_dataset, save_dataset


iterations = 100000
num_layers = 1000


# Define the kernel for benchmarking
def run_inference(operators, num_layers: int, required_iterations: int , input_tensor: torch.Tensor) -> torch.Tensor:
    for k in range(required_iterations):
        # Linearly access the convolutional layer from the pre-created list
        operator = operators[k % num_layers]
        
        # Apply the convolution operation
        output = operator(input_tensor)
    return output

# Define the range of iteration counts
min_iters = 1
max_iters = 100000
num_points = 150  # Adjust based on the number of test points you need

# Use a logarithmic scale to create the test points
test_points = np.logspace(np.log10(min_iters), np.log10(max_iters), num=num_points, dtype=int)

# Remove duplicates by converting to a set and back to a list, then sort
test_points = sorted(set(test_points))


iternumberlist= test_points#[200000,100000,50000, 10000, 5000, 1000, 500, 100, 50, 40, 30, 20, 10, 10, 10, 10 ,10 , 5 ,2 ,2 ,1 ,1 ,1,1] #500000, 100000,

input_size = (32, 64, 12, 12)

example_layer = nn.Conv2d(in_channels = 64 , out_channels = 128 , kernel_size=3, stride=1, padding=1)

operators = []
for _ in range(num_layers):
    layer = example_layer.cuda()
    if hasattr(example_layer, 'weight') and example_layer.weight is not None:
        if example_layer.weight.ndim < 2:
            init.uniform_(layer.weight, a=-0.1, b=0.1)
        else:
            init.xavier_uniform_(layer.weight)
    if hasattr(example_layer, 'bias') and example_layer.bias is not None:
        #init.xavier_uniform_(layer.bias)
        init.uniform_(layer.bias, a=-0.1, b=0.1)
    operators.append(layer)


ifmap = torch.randn(input_size).cuda()



for num in iternumberlist:



    # warmup_start_time = time.time()

    # # Warmup iterations, to avoid measuring the cold start of the gpu
    # for i in range(num):
    #     # Linearly access the convolutional layer from the pre-created list
    #     operator = operators[i % num_layers]
        
    #     # Apply the convolution operation
    #     output = operator(ifmap)

    # warmup_stop_time = time.time()

    # warmup_time = warmup_stop_time - warmup_start_time

    # print(num/warmup_time)





    # PyTorch Benchmark Timer
    num_repeats = 1  # Number of times to repeat the measurement
    timer = benchmark.Timer(
        stmt="run_inference(operators, num_layers,required_iterations, ifmap)",  # Statement to benchmark  # Setup the function and variables
        setup="from __main__ import run_inference",
        globals={
            "operators": operators,
            "required_iterations": num,
            "num_layers": num_layers,
            "ifmap": ifmap
        },
        num_threads=1,  # Number of threads to use
        label="Latency Measurement",
        sub_label="torch.utils.benchmark"
    )


    profile_result = timer.timeit(num_repeats)

    print(num,profile_result.mean)