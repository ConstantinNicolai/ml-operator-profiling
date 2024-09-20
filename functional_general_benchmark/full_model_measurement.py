import argparse
import importlib
import torch
from torchvision import models, transforms
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import re
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter
import ast
from collections import defaultdict
import pickle
import lzma
import yaml
import time
import math
import os
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new, process_log_file,get_latest_dataset_file, load_latest_dataset, save_dataset

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iterations = 10000


finishup = """
bg_pids=$(jobs -p)
for pid in $bg_pids; do
    kill $pid
done
"""


for entry in os.listdir('./../measurements'):
    with open('./../measurements/' + entry + '/summary.yml', 'r') as file:
        config = yaml.safe_load(file)

    config['input_size'] = tuple(config['input_size'])

    # Dynamically create variables
    for key, value in config.items():
        globals()[key] = value

    model = get_model_and_weights(model_name, weights_name)

    model = model.to(device)

    print(model_name, input_size)

    ifmap = torch.randn(input_size).cuda()

    warmup_start_time = time.time()

    # Warmup iterations, to avoid measuring the cold start of the gpu
    for i in range(math.ceil(iterations/4)):       
        # Apply the convolution operation
        output = model(ifmap)

    warmup_stop_time = time.time()

    warmup_time = warmup_stop_time - warmup_start_time

    time_per_iteration = warmup_time / math.ceil(iterations/4)

    required_iterations = int(30 / time_per_iteration)

    # print(required_iterations)

    # Create the startup command string with parameters
    startup = f"""
    nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits > current_temp.log &
    """


    # Starting the gpu stats logging in the background
    os.system(startup)


    # Start the timer
    start_time = time.time()


    # Run the convolution operation in a loop, accessing layers linearly
    for i in range(required_iterations):
        # Apply the convolution operation
        output = model(ifmap)

    # Stop the timer
    end_time = time.time()

    # Stopping the gpu stats logging running in the background
    os.system(finishup)

    # Calculate the time taken
    total_time = end_time - start_time
    # print(f"Total time for {required_iterations} iterations: {total_time:.4f} seconds")

    iterations, time_difference_seconds, time_per_iteration, filtered_mean_value2, std_value2, total_energy_joules, energy_per_iteration_in_milli_joule = process_log_file('current_temp.log', required_iterations)

    print(1000*time_per_iteration, "[ms]", energy_per_iteration_in_milli_joule, "mJ")




# inception_v3 (32, 3, 299, 299)
# (32, 3, 299, 299) 53.677364864864865 12414.93041224118
# resnet18 (32, 3, 224, 224)
# (32, 3, 224, 224) 13.419096209912537 3130.5669179301585
# resnet34 (32, 3, 224, 224)
# (32, 3, 224, 224) 19.704928664072634 4631.017081406672
# resnet34 (32, 3, 56, 56)
# (32, 3, 56, 56) 12.805813234384664 1175.2894863482675
