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

# # Check if CUDA is available and set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("hello hello")

# iterations = 20

# finishup = """
# bg_pids=$(jobs -p)
# for pid in $bg_pids; do
#     kill $pid
# done
# """

# custom_selected_models_list = ['resnet50_32,3,224,224', 'resnet18_32,3,299,299']

# # for entry in os.listdir('./../measurements/RTX2080TI'):
# for entry in custom_selected_models_list:
#     with open('./../measurements/RTX2080TI/' + entry + '/summary.yml', 'r') as file:
#         config = yaml.safe_load(file)

#     print(" helllo sdfasfsd")

#     config['input_size'] = tuple(config['input_size'])

#     # # Dynamically create variables
#     # for key, value in config.items():
#     #     globals()[key] = value

#     model = get_model_and_weights(config['model_name'], config['weights_name'])

#     model = model.to(device)

#     print(config['model_name'], config['input_size'])

#     ifmap = torch.randn(config['input_size']).cuda()

#     warmup_start_time = time.time()

#     # Warmup iterations, to avoid measuring the cold start of the gpu
#     with torch.no_grad():
#         for i in range(math.ceil(iterations/4)):       
#             # Apply the convolution operation
#             output = model(ifmap)

#     warmup_stop_time = time.time()

#     warmup_time = warmup_stop_time - warmup_start_time

#     print(warmup_time)

    # time_per_iteration = warmup_time / math.ceil(iterations/4)

    # required_iterations = int(30 / time_per_iteration)

    # # print(required_iterations)

    # # Create the startup command string with parameters
    # startup = f"""
    # nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits > current_temp_full_new_RTX2080TI.log &
    # """
    
    # # Starting the gpu stats logging in the background
    # os.system(startup)


    # # Start the timer
    # start_time = time.time()


    # # Run the convolution operation in a loop, accessing layers linearly
    # with torch.no_grad():
    #     for i in range(required_iterations):
    #         # Apply the convolution operation
    #         output = model(ifmap)

    # # Stop the timer
    # end_time = time.time()

    # # Stopping the gpu stats logging running in the background
    # os.system(finishup)

    # # Calculate the time taken
    # total_time = end_time - start_time

    # (
    #     iterations, 
    #     time_difference_seconds, 
    #     time_per_iteration,
    #     filtered_mean_value2, 
    #     filtered_std_value2, 
    #     total_energy_joules,
    #     energy_per_iteration_in_milli_joule, 
    #     total_energy_joules_error,
    #     energy_per_iteration_in_milli_joule_error,
    #     energy_per_iteration_in_milli_joule_std
    # ) = process_log_file('current_temp_full_new_RTX2080TI.log', required_iterations)

    # print(
    #     1000*time_per_iteration,"[ms]", 
    #     energy_per_iteration_in_milli_joule, "mJ",
    #     energy_per_iteration_in_milli_joule_error, "mJ",
    #     energy_per_iteration_in_milli_joule_std, "mJ")