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
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new, process_log_file,get_latest_dataset_file, load_latest_dataset, save_dataset


iterations = 100000
num_layers = 500

h = 6


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
        
    tuple_str = "_".join(map(str, input_size))
    filename = f"{model_name}_{tuple_str}.pkl.xz"

    if done == False:
        with lzma.open('./../measurements/' + entry + '/' + filename + '_filtered') as file_:
            saved_dict = pickle.load(file_)
        

        list_attemps = list(saved_dict.items())

        if list_attemps:

            new_measurements = []

            number_of_unique_operators_in_model = len(list_attemps)

            for h in range(number_of_unique_operators_in_model):


                input_size = list_attemps[h][0][2]


                example_layer = list_attemps[h][1][0]


                operators = []
                for _ in range(num_layers):
                    layer = list_attemps[h][1][0].cuda()
                    if hasattr(example_layer, 'weight') and example_layer.weight is not None:
                        init.xavier_uniform_(layer.weight)
                    if hasattr(example_layer, 'weight') and example_layer.bias is not None:
                        init.xavier_uniform_(layer.bias)
                    operators.append(layer)


                ifmap = torch.randn(input_size).cuda()

                warmup_start_time = time.time()

                # Warmup iterations, to avoid measuring the cold start of the gpu
                for i in range(math.ceil(iterations/4)):
                    # Linearly access the convolutional layer from the pre-created list
                    operator = operators[i % num_layers]
                    
                    # Apply the convolution operation
                    output = operator(ifmap)

                warmup_stop_time = time.time()

                warmup_time = warmup_stop_time - warmup_start_time

                time_per_iteration = warmup_time / math.ceil(iterations/4)

                required_iterations = int(30 / time_per_iteration)

                # print(required_iterations)

                # Create the startup command string with parameters
                startup = f"""
                nvidia-smi -lms=1 --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total,pstate --format=csv,noheader,nounits > current_temp.log &
                """


                # Starting the gpu stats logging in the background
                os.system(startup)


                # Start the timer
                start_time = time.time()


                # Run the convolution operation in a loop, accessing layers linearly
                for i in range(required_iterations):
                    # Linearly access the convolutional layer from the pre-created list
                    operator = operators[i % num_layers]
                    
                    # Apply the convolution operation
                    output = operator(ifmap)

                # Stop the timer
                end_time = time.time()

                # Stopping the gpu stats logging running in the background
                os.system(finishup)

                # Calculate the time taken
                total_time = end_time - start_time
                # print(f"Total time for {required_iterations} iterations: {total_time:.4f} seconds")

                iterations, time_difference_seconds, time_per_iteration, filtered_mean_value2, std_value2, total_energy_joules, energy_per_iteration_in_milli_joule = process_log_file('current_temp.log', required_iterations)

                print(example_layer, input_size, time_per_iteration, energy_per_iteration_in_milli_joule)

                new_measurements.append((example_layer, input_size, time_per_iteration, energy_per_iteration_in_milli_joule))


            DATASET_DIR = "dataset_history/"

            # Ensure the dataset directory exists
            os.makedirs(DATASET_DIR, exist_ok=True)

            # Example: Load, append, and save the dataset
            dataset = load_latest_dataset()

            dataset.extend(new_measurements)  # Append new data
            save_dataset(dataset)


        





##############################################################################
# approach I prompted 


# import torch
# import torch.nn as nn
# from collections import defaultdict
# from statistics import mean

# # Create example dataset
# dataset = [
#     (nn.Conv2d(3, 64, kernel_size=3), (64, 3, 32, 32), 0.5, 0.01, 15.2, 0.2),
#     (nn.Conv2d(3, 64, kernel_size=3), (64, 3, 32, 32), 0.6, 0.02, 14.8, 0.3),
#     (nn.Linear(64, 10), (64, 64), 0.7, 0.03, 10.5, 0.1),
#     (nn.Linear(64, 10), (64, 64), 0.8, 0.04, 10.6, 0.2)
# ]

# # Helper function to create a key from the layer type, extra_repr, and input shape
# def layer_to_key(layer, input_shape):
#     # Use layer type (_get_name) and its configuration (extra_repr), along with input shape
#     layer_name = layer._get_name()
#     layer_repr = layer.extra_repr()  # Describes the layer's configuration
#     return (layer_name, layer_repr, input_shape)

# # Group entries by (layer_key, input_tuple)
# grouped_data = defaultdict(list)

# for layer, input_tuple, time, delta_time, energy, delta_energy in dataset:
#     # Create a unique key based on the layer type, config (extra_repr), and input shape
#     layer_key = layer_to_key(layer, input_tuple)
    
#     # Append measurements to the corresponding group
#     grouped_data[layer_key].append((time, delta_time, energy, delta_energy))

# # Example: Print out the groups
# for key, measurements in grouped_data.items():
#     layer_name, layer_repr, input_shape = key
#     print(f"Layer: {layer_name} ({layer_repr}), Input Shape: {input_shape}, Measurements: {measurements}")

# # Now, you can process the measurements, for example, by taking the mean of each group
# processed_data = []
# for key, measurements in grouped_data.items():
#     # Unpack the measurements
#     times, delta_times, energies, delta_energies = zip(*measurements)
    
#     # Calculate means (or apply other statistical methods)
#     mean_time = mean(times)
#     mean_delta_time = mean(delta_times)
#     mean_energy = mean(energies)
#     mean_delta_energy = mean(delta_energies)
    
#     # Store the processed data (layer_key, input_tuple, mean_time, mean_delta_time, mean_energy, mean_delta_energy)
#     processed_data.append((key[0], key[2], mean_time, mean_delta_time, mean_energy, mean_delta_energy))

# # Example: Print out the processed data
# for item in processed_data:
#     layer_name, input_shape, mean_time, mean_delta_time, mean_energy, mean_delta_energy = item
#     print(f"Layer: {layer_name}, Input: {input_shape}, Mean Time: {mean_time}, Mean Energy: {mean_energy}")


##############################################################################################



