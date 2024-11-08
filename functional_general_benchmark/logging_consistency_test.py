import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.init as init
import pickle
import lzma
import yaml
import os
import torch.nn.init as init
import time
import math
from datetime import datetime
import uncertainties
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new, process_log_file,get_latest_dataset_file, load_latest_dataset, save_dataset
import subprocess
import time
import os
import signal

processes = []

iterations = 100000
num_layers = 1000



finishup = """
bg_pids=$(jobs -p)
for pid in $bg_pids; do
    kill $pid
done
"""


# def


#     for j in range(required_iterations):  # Use `j` again here
#         operator = operators[j % num_layers]
#         output = operator(ifmap)



processes = []

# with lzma.open('outlier_rerun_RTX2080TI') as file_:
#     saved_dict = pickle.load(file_)


## Running with tensor cores available. If comment and state mismatch your data is wrong
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# outlier_list = list(saved_dict.items())

input_size = (32, 128, 8, 8)

example_layer = nn.Conv2d(in_channels = 128 , out_channels = 256 , kernel_size=3, stride=1, padding=1)

new_measurements = []

outside_logging_command = (
    f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
    f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
    f"> current_continousA30.log"
)

outside_log = subprocess.Popen(outside_logging_command, shell=True, preexec_fn=os.setsid)


processes = []  # Ensure this list is initialized before the main loop



input_size = (32, 256, 12, 12)
example_layer = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

operators = []
for _ in range(num_layers):
    layer = example_layer.cuda()
    if hasattr(layer, 'weight') and layer.weight is not None:
        if layer.weight.ndim < 2:
            init.uniform_(layer.weight, a=-0.1, b=0.1)
        else:
            init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        init.uniform_(layer.bias, a=-0.1, b=0.1)
    operators.append(layer)

ifmap = torch.randn(input_size).cuda()



# Warmup loop
warmup_start_time = time.time()
for j in range(25000):  # Renamed inner loop variable to `j`
    operator = operators[j % num_layers]
    output = operator(ifmap)
warmup_stop_time = time.time()
warmup_time = warmup_stop_time - warmup_start_time

# Determine required iterations
time_per_iteration = warmup_time / 25000
required_iterations = int(30 / time_per_iteration)

for i in range(5):

    # Start GPU stats logging and save the process
    startup_command = (
        f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
        f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
        f"> current_temp_A30_sleep{i}.log"
    )

    time.sleep(10)

    process = subprocess.Popen(startup_command, shell=True, preexec_fn=os.setsid)
    processes.append(process)

    # Run main loop
    start_time = time.time()
    for j in range(required_iterations):  # Use `j` again here
        operator = operators[j % num_layers]
        output = operator(ifmap)
    end_time = time.time()

    # Stop GPU stats logging for the latest process
    os.killpg(os.getpgid(processes[-1].pid), signal.SIGTERM)  # Use processes[-1] to get the last process

    # Calculate and print total time
    total_time = end_time - start_time
    print(f"Total time for {required_iterations} iterations: {total_time:.4f} seconds")

    log_filename = f"current_temp_RTX2080TI_sleep{i}.log"


    (
        iterations, 
        time_difference_seconds, 
        time_per_iteration,
        filtered_mean_value2, 
        filtered_std_value2, 
        total_energy_joules,
        energy_per_iteration_in_milli_joule, 
        total_energy_joules_error,
        energy_per_iteration_in_milli_joule_error,
        energy_per_iteration_in_milli_joule_std
    ) = process_log_file(log_filename, required_iterations)

    
    # print(
    #     example_layer,
    #     input_size,
    #     time_per_iteration,
    #     energy_per_iteration_in_milli_joule,
    #     energy_per_iteration_in_milli_joule_error,
    #     energy_per_iteration_in_milli_joule_std,
    #     iterations, 
    #     time_difference_seconds, 
    #     filtered_mean_value2, 
    #     filtered_std_value2, 
    #     total_energy_joules, 
    #     total_energy_joules_error
    # )

    new_measurements.append((
        example_layer,
        input_size,
        time_per_iteration,
        energy_per_iteration_in_milli_joule,
        energy_per_iteration_in_milli_joule_error,
        energy_per_iteration_in_milli_joule_std,
        iterations, 
        time_difference_seconds, 
        filtered_mean_value2, 
        filtered_std_value2, 
        total_energy_joules, 
        total_energy_joules_error
    ))

os.killpg(os.getpgid(outside_log.pid), signal.SIGTERM)

for item in new_measurements:
    print(item)