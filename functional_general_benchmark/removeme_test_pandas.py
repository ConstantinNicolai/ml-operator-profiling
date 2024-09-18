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



# iterations, time_difference_seconds, time_per_iteration, filtered_mean_value2, std_value2, total_energy_joules, energy_per_iteration_in_milli_joule = process_log_file('../working_example_pytorch_operator_loop/logs/conv2d_3in_64out_7k_2s_3p_32b_224ifm_28837iter.log', 7000)


# print(time_per_iteration, energy_per_iteration_in_milli_joule)



# for entry in os.listdir('./../measurements'):
#     with open('./../measurements/' + entry + '/summary.yml', 'r') as file:
#         config = yaml.safe_load(file)

#     config['input_size'] = tuple(config['input_size'])

#     # Dynamically create variables
#     for key, value in config.items():
#         globals()[key] = value
        
#     tuple_str = "_".join(map(str, input_size))
#     filename = f"{model_name}_{tuple_str}.pkl.xz"

#     if done == False:
#         with lzma.open('./../measurements/' + entry + '/' + filename + '_filtered') as file_:
#             saved_dict = pickle.load(file_)
        

#         list_attemps = list(saved_dict.items())

#         print("#############################################################################")

#         if list_attemps:
#             number_of_unique_operators_in_model = len(list_attemps)

#             for h in range(number_of_unique_operators_in_model):

#                 print(list_attemps[h])


#                # input_size = list_attemps[h][0][2]


hululayer = nn.Linear(4, 4, bias=True)

gglayer = nn.Conv2d(3,6,(1,1), bias=True)


# print(hululayer)
# print(hululayer.bias)

# # Initialize bias with small uniform values
# init.uniform_(hululayer.bias, a=-0.1, b=0.1)

# print(hululayer.bias)


print('###################')

print(gglayer.bias)

# Initialize bias with small uniform values
init.uniform_(gglayer.bias, a=-0.1, b=0.1)

print(gglayer.bias)