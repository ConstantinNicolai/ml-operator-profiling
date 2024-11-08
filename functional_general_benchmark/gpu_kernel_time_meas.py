import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import torch.nn.init as init
import time
import subprocess
import os
import signal
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new, process_log_file,get_latest_dataset_file, load_latest_dataset, save_dataset




# Define the kernel for benchmarking
def run_inference(operators, num_layers: int, required_iterations: int , input_tensor: torch.Tensor) -> torch.Tensor:
    for k in range(required_iterations):
        # Linearly access the convolutional layer from the pre-created list
        operator = operators[k % num_layers]
        
        # Apply the convolution operation
        output = operator(input_tensor)
    return output


outside_logging_command = (
    f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
    f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
    f"> current_continousA30.log"
)

processes = []

# Create a simple model and input tensor
input_shape = (1, 3, 224, 224)  # For example, a single image with 3 channels and 224x224 resolution
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_layers = 500
ifmap = torch.randn(input_shape).cuda()


example_layer = nn.Conv2d(in_channels=input_shape[1], out_channels=256, kernel_size=(5, 5)).cuda()



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


outside_log = subprocess.Popen(outside_logging_command, shell=True, preexec_fn=os.setsid)

warmup_start_time = time.time()

# Warmup iterations, to avoid measuring the cold start of the gpu
for i in range(25000):
    # Linearly access the convolutional layer from the pre-created list
    operator = operators[i % num_layers]
    
    # Apply the convolution operation
    output = operator(ifmap)

warmup_stop_time = time.time()

warmup_time = warmup_stop_time - warmup_start_time

time_per_iteration = warmup_time / 25000

required_iterations = int(20 / time_per_iteration)

print(required_iterations)

for i in range(2):

    # Start GPU stats logging and save the process
    startup_command = (
        f"nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,"
        f"power.draw,memory.used,memory.total --format=csv,noheader,nounits "
        f"> current_temp_A30_sleep{i}.log"
    )

    time.sleep(10)

    # PyTorch Benchmark Timer
    # num_repeats = 1  # Number of times to repeat the measurement
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

    process = subprocess.Popen(startup_command, shell=True, preexec_fn=os.setsid)
    processes.append(process)

    profile_result = timer.blocked_autorange(min_run_time=0)

    # Stop GPU stats logging for the latest process
    os.killpg(os.getpgid(processes[i].pid), signal.SIGTERM)  # Use processes[-1] to get the last process

    end_sub = time.time()

    subprocess_outside_time= end_sub-start_sub

    print("subprocess outside measured time: ", subprocess_outside_time)
    print(f"Elapsed time subprocess: {subprocess_outside_time} seconds")

    # # Run the benchmark and output the result
    # profile_result = timer.timeit(num_repeats)

    # Calculate and print total time
    print(f"Latency: {profile_result.mean}s")
    print(f"Time per iteration: {(profile_result.mean/required_iterations) * 1000:.5f}ms")

    log_filename = f"current_temp_A30_sleep{i}.log"


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

    
    print(
        time_per_iteration,
        time_difference_seconds, 
    )

os.killpg(os.getpgid(outside_log.pid), signal.SIGTERM)