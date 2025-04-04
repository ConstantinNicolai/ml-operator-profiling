import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.init as init
import torch.optim as optim
import torchvision.models as models
import pickle
import lzma
import yaml
import os
import time
import subprocess
import signal
import argparse
import math
import statistics
from datetime import datetime
import torch.utils.benchmark as benchmark
from utils import (get_model_and_weights, extract_layer_info, parse_model_and_weights, 
process_model, forward_hook_new, process_log_file,get_latest_dataset_file, load_latest_dataset,
save_dataset, get_num_classes)


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iterations = 20

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Set GPU type for benchmarking.")
parser.add_argument("--gpu", type=str, required=True, help="Specify the GPU type (e.g., A30, V100, etc.)")
parser.add_argument("--rundur", type=int, required=True, help="Set duration per benchmark run of N iteration")
parser.add_argument("--runnr", type=int, required=True, help="Set number of runs")

# Parse arguments
args = parser.parse_args()
gpu = args.gpu
rundur = args.rundur
runnr = args.runnr

if gpu == "A30_no_tc":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False



# Define the kernel for benchmarking
def run_training(operator, optimizer, loss_fn, required_iterations: int, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
    for k in range(required_iterations):
        optimizer.zero_grad()  # Reset gradients
        output = operator(input_tensor)  # Forward pass
        loss = loss_fn(output, target_tensor)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        torch.cuda.synchronize()
    return loss.item()

outside_log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outside_logging_command = (
    f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
    f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
    f"> continous_logs/current_continuous_{gpu}_{outside_log_timestamp}.log"
)

processes = []

outside_log = subprocess.Popen(outside_logging_command, shell=True, preexec_fn=os.setsid)


new_measurements = []

meas_dir_path = f"./../predictions/A30"
meas_dir_path_to_be_added = f"./../predictions/A30/"

for entry in os.listdir(meas_dir_path):
    with open(meas_dir_path_to_be_added + entry + '/summary.yml', 'r') as file:
        config = yaml.safe_load(file)

    try:

        config['input_size'] = tuple(config['input_size'])
            
        model = get_model_and_weights(config['model_name'], config['weights_name'])

        model = model.to(device)

        model.train()

        print(config['model_name'], config['input_size'])

        if isinstance(model, models.Inception3):
            print("skipping inception")
            continue

        input_size = config['input_size']

        batch_size = input_size[0]

        ifmap = torch.randn(config['input_size']).cuda()

        num_classes = get_num_classes(model)

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        target = torch.randint(0, num_classes, (batch_size,), device="cuda")

        required_iterations = 16


        # PyTorch Benchmark Timer
        num_repeats = 1  # Number of times to repeat the measurement
        timer = benchmark.Timer(
            stmt="run_training(operator, optimizer, loss_fn, required_iterations, ifmap, target)",
            setup="from __main__ import run_training",
            globals={
                "operator": model,
                "optimizer": optimizer,
                "loss_fn": loss_fn,
                "required_iterations": required_iterations,
                "ifmap": ifmap,
                "target": target  # Assuming target tensor is defined
            },
            num_threads=1,
            label="Training Latency Measurement",
            sub_label="torch.utils.benchmark"
        )

        warmup_result = timer.blocked_autorange(callback=None, min_run_time=4)


        print("warmup complete", config['model_name'], config['input_size'])



        startup_command = (
            f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
            f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
            f"> current_full_model_{gpu}.log"
        )



        start_sub = time.time()
        operation_start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        process = subprocess.Popen(startup_command, shell=True, preexec_fn=os.setsid)
        processes.append(process)

        #Actual benchmarking call

        # profile_result = timer.timeit(num_repeats)
        profile_result = timer.blocked_autorange(callback=None, min_run_time=rundur * runnr)

        # Stop GPU stats logging for the latest process
        os.killpg(os.getpgid(processes[-1].pid), signal.SIGTERM)  # Use processes[-1] to get the last process

        operation_stop_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        end_sub = time.time()

        subprocess_outside_time= end_sub-start_sub


        # print(f"Elapsed time subprocess: {subprocess_outside_time} seconds")
        # # Calculate and print total time
        # print(f"Latency, should be 1/3 of elapsed time: {profile_result.mean}s")
        model = model.cpu()
        torch.cuda.empty_cache()


        log_running_atm = f"current_full_model_{gpu}.log"

        (
            iterations, 
            time_difference_seconds, 
            old_time_per_iteration,
            filtered_power_mean, 
            filtered_power_std, 
            total_energy_joules,
            energy_per_iteration_in_milli_joule,
            total_energy_joules_error,
            energy_per_iteration_in_milli_joule_error,
            energy_per_iteration_in_milli_joule_std
        ) = process_log_file(log_running_atm, len(profile_result.times)*required_iterations)


        stddev_time = statistics.stdev(profile_result.times)
        new_time_per_iteration = profile_result.mean/required_iterations
        new_time_per_iteration_std = stddev_time/required_iterations
        new_energy_per_iteration_in_milli_joule = new_time_per_iteration*filtered_power_mean*1000
        relative_error_time_per_iteration = new_time_per_iteration_std/new_time_per_iteration
        relative_error_power = filtered_power_std/filtered_power_mean
        new_energy_per_iteration_in_milli_joule_std = 1000 * new_time_per_iteration*filtered_power_mean *math.sqrt(relative_error_time_per_iteration**2 + relative_error_power**2)
        total_runtime = sum(profile_result.times)
        total_energy_joules = (new_energy_per_iteration_in_milli_joule / 1000) * required_iterations
        total_energy_joules_std = (new_energy_per_iteration_in_milli_joule_std / 1000) * required_iterations


        print(
            model.__class__.__name__,
            input_size,
            new_time_per_iteration,
            new_energy_per_iteration_in_milli_joule,
            energy_per_iteration_in_milli_joule_error,
            new_energy_per_iteration_in_milli_joule_std,
            iterations, 
            total_runtime,
            filtered_power_mean, 
            filtered_power_std, 
            total_energy_joules, 
            total_energy_joules_std,
            old_time_per_iteration,
            operation_start_datetime,
            operation_stop_datetime,
            new_time_per_iteration_std
        )


        new_measurements.append((
            model,
            input_size,
            new_time_per_iteration,
            new_energy_per_iteration_in_milli_joule,
            energy_per_iteration_in_milli_joule_error,
            new_energy_per_iteration_in_milli_joule_std,
            iterations, 
            total_runtime,
            filtered_power_mean, 
            filtered_power_std, 
            total_energy_joules, 
            total_energy_joules_std,
            old_time_per_iteration,
            operation_start_datetime,
            operation_stop_datetime,
            new_time_per_iteration_std,
            config['model_name']
        ))
    except torch.cuda.OutOfMemoryError:
        print("CUDA Out of Memory Error: Model is too large for the GPU.")
        torch.cuda.empty_cache()  # Free up memory



DATASET_DIR = f"datasets_fullmodel_validation/dataset_history_{gpu}/"

# Ensure the dataset directory exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Example: Load, append, and save the dataset
dataset = load_latest_dataset(DATASET_DIR)

dataset.extend(new_measurements)  # Append new data
save_dataset(dataset, DATASET_DIR)