import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.init as init
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
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new, process_log_file,get_latest_dataset_file, load_latest_dataset, save_dataset


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
def run_inference(operators, num_layers: int, required_iterations: int , input_tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        for k in range(required_iterations):
            # Linearly access the convolutional layer from the pre-created list
            operator = operators[k % num_layers].eval()
            
            # Apply the convolution operation
            output = operator(input_tensor)
    return output


# Define the kernel for benchmarking
def run_training(operators, num_layers: int, required_iterations: int, input_tensor: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    #input_tensor.requires_grad = True  # Enable gradient computation
    input_tensor = input_tensor + 0

    for k in range(required_iterations):
        # Linearly access the convolutional layer from the pre-created list
        operator = operators[k % num_layers].train()

        # Apply the convolution operation
        output = operator(input_tensor)

        output.backward(gradient)

        # Reset input tensor gradients after each backward pass
        input_tensor.grad = None


        # Zero gradients collectively when cycling back to the start of the operator list
        if (k + 1) % num_layers == 0:
            with torch.no_grad():
                # Collectively zero out all gradients for all operators
                for param in [p for op in operators for p in op.parameters() if p.grad is not None]:
                    param.grad.zero_()

    return output

outside_log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outside_logging_command = (
    f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
    f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
    f"> continous_logs/current_continuous_{gpu}_{outside_log_timestamp}.log"
)

num_layers = 500
processes = []

outside_log = subprocess.Popen(outside_logging_command, shell=True, preexec_fn=os.setsid)


meas_dir_path = f"./../measurements/{gpu}"
meas_dir_path_to_be_added = f"./../measurements/{gpu}/"

for entry in os.listdir(meas_dir_path):
    with open(meas_dir_path_to_be_added + entry + '/summary.yml', 'r') as file:
        config = yaml.safe_load(file)

    config['input_size'] = tuple(config['input_size'])
        
    tuple_str = "_".join(map(str, config['input_size']))
    filename = f"{config['model_name']}_{tuple_str}.pkl.xz"

    if config['done'] == True:
        print("done flag already set to true, for rerun reset to false")
    if config['done'] == False:
        with lzma.open(meas_dir_path_to_be_added + entry + '/' + filename + '_filtered') as file_:
            saved_dict = pickle.load(file_)
        

        list_attemps = list(saved_dict.items())

        if list_attemps:

            new_measurements = []

            number_of_unique_operators_in_model = len(list_attemps)

            for h in range(number_of_unique_operators_in_model):

                startup_command = (
                    f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
                    f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
                    f"> current_{gpu}.log"
                )


                input_size = list_attemps[h][0][2]


                example_layer1 = list_attemps[h][1][0]

                print(example_layer1)

                example_layer = example_layer1

                if hasattr(example_layer, 'inplace'):

                    example_layer.inplace = False




                


                operators = []
                for _ in range(num_layers):
                    layer = list_attemps[h][1][0].cuda()
                    if hasattr(example_layer, 'weight') and example_layer.weight is not None:
                        if example_layer.weight.ndim < 2:
                            init.uniform_(layer.weight, a=-0.1, b=0.1)
                        else:
                            init.xavier_uniform_(layer.weight)
                    if hasattr(example_layer, 'bias') and example_layer.bias is not None:
                        #init.xavier_uniform_(layer.bias)
                        init.uniform_(layer.bias, a=-0.1, b=0.1)
                    operators.append(layer)


                ifmap = torch.randn(input_size, device='cuda', requires_grad=True)

                warmup_start_time = time.time()

                # Warmup iterations, to avoid measuring the cold start of the gpu
                with torch.inference_mode():
                    for i in range(10000):
                        # Linearly access the convolutional layer from the pre-created list
                        operator = operators[i % num_layers]
                        
                        # Apply the convolution operation
                        output = operator(ifmap)

                gradient = torch.randn_like(output)

                warmup_stop_time = time.time()

                warmup_time = warmup_stop_time - warmup_start_time

                time_per_iteration = warmup_time / 10000

                required_iterations = int(rundur / time_per_iteration)

                # PyTorch Benchmark Timer
                num_repeats = 1  # Number of times to repeat the measurement
                timer = benchmark.Timer(
                    stmt="run_inference(operators, num_layers,required_iterations, ifmap)",  # Statement to benchmark  # Setup the function and variables
                    setup="from __main__ import run_inference",
                    globals={
                        "operators": operators,
                        "required_iterations": required_iterations,
                        "num_layers": num_layers,
                        "ifmap": ifmap
                    },
                    num_threads=1,  # Number of threads to use
                    label="Latency Measurement",
                    sub_label="torch.utils.benchmark"
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


                end_sub = time.time()

                subprocess_outside_time= end_sub-start_sub


                # print(f"Elapsed time subprocess: {subprocess_outside_time} seconds")
                # # Calculate and print total time
                # print(f"Latency, should be 1/3 of elapsed time: {profile_result.mean}s")



                log_running_atm = f"current_{gpu}.log"

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


                ##########################################################################################################
                #Do everything again to measure training performance

                startup_command1 = (
                    f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
                    f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
                    f"> current_train_{gpu}.log"
                )




                warmup_start_time = time.time()
                ifmap = ifmap + 0

                # Warmup iterations, to avoid measuring the cold start of the gpu
                for k in range(10000):
                    # Linearly access the convolutional layer from the pre-created list
                    operator = operators[k % num_layers]

                    # Apply the convolution operation
                    output = operator(ifmap)

                    # Use a simple scalar loss (sum of the output) for benchmarking
                    output.backward(gradient)

                    ifmap.grad = None

                    # Zero gradients collectively when cycling back to the start of the operator list
                    if (k + 1) % num_layers == 0:
                        with torch.no_grad():
                            # Collectively zero out all gradients for all operators
                            for param in [p for op in operators for p in op.parameters() if p.grad is not None]:
                                param.grad.zero_()


                warmup_stop_time = time.time()

                warmup_time = warmup_stop_time - warmup_start_time

                time_per_iteration = warmup_time / 10000

                required_iterations_train = int(rundur / time_per_iteration)



                # PyTorch Benchmark Timer
                num_repeats = 1  # Number of times to repeat the measurement
                timer = benchmark.Timer(
                    stmt="run_training(operators, num_layers,required_iterations, ifmap, gradient)",  # Statement to benchmark  # Setup the function and variables
                    setup="from __main__ import run_training",
                    globals={
                        "operators": operators,
                        "required_iterations": required_iterations_train,
                        "num_layers": num_layers,
                        "ifmap": ifmap,
                        "gradient" : gradient
                    },
                    num_threads=1,  # Number of threads to use
                    label="Latency Measurement",
                    sub_label="torch.utils.benchmark"
                )


                start_sub = time.time()

                process = subprocess.Popen(startup_command1, shell=True, preexec_fn=os.setsid)
                processes.append(process)

                #Actual benchmarking call

                # profile_result = timer.timeit(num_repeats)
                profile_result_train = timer.blocked_autorange(callback=None, min_run_time=rundur * runnr)

                # Stop GPU stats logging for the latest process
                os.killpg(os.getpgid(processes[-1].pid), signal.SIGTERM)  # Use processes[-1] to get the last process

                operation_stop_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                end_sub = time.time()

                subprocess_outside_time= end_sub-start_sub


                # print(f"Elapsed time subprocess: {subprocess_outside_time} seconds")
                # # Calculate and print total time
                # print(f"Latency, should be 1/3 of elapsed time: {profile_result.mean}s")

                log_running_atm = f"current_train_{gpu}.log"


                (
                    iterations_train, 
                    time_difference_seconds_train, 
                    old_time_per_iteration_train,
                    filtered_power_mean_train, 
                    filtered_power_std_train, 
                    total_energy_joules_train,
                    energy_per_iteration_in_milli_joule_train,
                    total_energy_joules_error_train,
                    energy_per_iteration_in_milli_joule_error_train,
                    energy_per_iteration_in_milli_joule_std_train
                ) = process_log_file(log_running_atm, len(profile_result_train.times)*required_iterations_train)


                stddev_time_train = statistics.stdev(profile_result_train.times)
                new_time_per_iteration_train = profile_result_train.mean/required_iterations_train
                new_time_per_iteration_std_train = stddev_time_train/required_iterations_train
                new_energy_per_iteration_in_milli_joule_train = new_time_per_iteration_train*filtered_power_mean_train*1000
                relative_error_time_per_iteration_train = new_time_per_iteration_std_train/new_time_per_iteration_train
                relative_error_power_train = filtered_power_std_train/filtered_power_mean_train
                new_energy_per_iteration_in_milli_joule_std_train = 1000 * new_time_per_iteration_train*filtered_power_mean_train *math.sqrt(relative_error_time_per_iteration_train**2 + relative_error_power_train**2)
                total_runtime_train = sum(profile_result_train.times)
                total_energy_joules_train = (new_energy_per_iteration_in_milli_joule_train / 1000) * required_iterations_train
                total_energy_joules_std_train = (new_energy_per_iteration_in_milli_joule_std_train / 1000) * required_iterations_train


                ##########################################################################################################


                print(
                    example_layer1,
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
                    new_time_per_iteration_train,
                    new_time_per_iteration_std_train,
                    new_energy_per_iteration_in_milli_joule_train,
                    new_energy_per_iteration_in_milli_joule_std_train,
                    total_runtime_train,
                    filtered_power_mean_train,
                    filtered_power_std_train,
                    total_energy_joules_train, 
                    total_energy_joules_std_train
                )

                new_measurements.append((
                    example_layer1,
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
                    new_time_per_iteration_train,
                    new_time_per_iteration_std_train,
                    new_energy_per_iteration_in_milli_joule_train,
                    new_energy_per_iteration_in_milli_joule_std_train,
                    total_runtime_train,
                    filtered_power_mean_train,
                    filtered_power_std_train,
                    total_energy_joules_train, 
                    total_energy_joules_std_train
                ))



            DATASET_DIR = f"datasets_train/dataset_history_{gpu}/"

            # Ensure the dataset directory exists
            os.makedirs(DATASET_DIR, exist_ok=True)

            # Example: Load, append, and save the dataset
            dataset = load_latest_dataset(DATASET_DIR)

            dataset.extend(new_measurements)  # Append new data
            save_dataset(dataset, DATASET_DIR)
            
    config['done'] = True

    with open(meas_dir_path_to_be_added + entry + '/summary.yml', 'w') as file:
        yaml.safe_dump(config, file)


os.killpg(os.getpgid(outside_log.pid), signal.SIGTERM)