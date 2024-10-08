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
import os
import glob
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model


# Define the forward hook function
def forward_hook_new(module, input, output):
    if not len(list(module.children())):
        # Extract the input shape (assuming input is a tuple)
        input_shape = tuple(input[0].size())

        hook_handles[module].remove()

        # Create a key based on module name, extra_repr, and input shape
        key = (module._get_name(), module.extra_repr(), input_shape)

        # Check if the key exists in the dict
        if opus_magnum_dict[key][0] is None:
            # If it's the first occurrence, store the module object and set the count to 1
            opus_magnum_dict[key] = [module, 1]
        else:
            # If we've seen this combination before, increment the count
            opus_magnum_dict[key][1] += 1



for summary_file in glob.glob('./../measurements/*/*/summary.yml'):
    HW_dir = os.path.dirname(summary_file)
    with open(summary_file, 'r') as file:
        config = yaml.safe_load(file)


    config['input_size'] = tuple(config['input_size'])

    if config['done'] == True:
        print("done flag already set to true, reset to false for rerun")
    if config['done'] == False:
        model = get_model_and_weights(config['model_name'], config['weights_name'])


        ##########################################################################


        # Generate random input data
        input_data = torch.randn(*input_size)


        # # Register the forward hook only for leaf nodes
        # for module in model.modules():
        #     handle = module.register_forward_hook(forward_hook_new)

        hook_handles = {}

        # Loop through the modules and register hooks
        for module in model.modules():
            handle = module.register_forward_hook(forward_hook_new)
            hook_handles[module] = handle


        # Create the defaultdict to store the module (first occurrence) and the count
        opus_magnum_dict = defaultdict(lambda: [None, 0])  # {key: [first_module_object, count]}


        output = model(input_data)


        opus_magnum_dict = dict(opus_magnum_dict)

        # for key, value in opus_magnum_dict.items():
        #     print(f'{key}: {value}')


        tuple_str = "_".join(map(str, input_size))

        # Format the filename using both variables 
        filename = f"{config['model_name']}_{tuple_str}.pkl.xz"

        print(HW_dir + '/' + filename)

        with lzma.open(HW_dir + '/' + filename, "wb") as file_:
            pickle.dump(opus_magnum_dict, file_)

        config['done'] = False
        config['input_size'] = list(config['input_size'])

        with open(summary_file, 'w') as file:
            yaml.safe_dump(config, file)

        




# with open('./../measurements/resnet18_32,3,224,224/summary.yml', 'r') as file:
#     config = yaml.safe_load(file)


# config['input_size'] = tuple(config['input_size'])

# # Dynamically create variables
# for key, value in config.items():
#     globals()[key] = value



# model = get_model_and_weights(model_name, weights_name)


# ##########################################################################


# # Generate random input data
# input_data = torch.randn(*input_size)


# # Register the forward hook only for leaf nodes
# for module in model.modules():
#     # if len(list(module.children())) == 0:
#     module.register_forward_hook(forward_hook_new)


# # Create the defaultdict to store the module (first occurrence) and the count
# opus_magnum_dict = defaultdict(lambda: [None, 0])  # {key: [first_module_object, count]}


# output = model(input_data)


# opus_magnum_dict = dict(opus_magnum_dict)

# # for key, value in opus_magnum_dict.items():
# #     print(f'{key}: {value}')


# tuple_str = "_".join(map(str, input_size))

# # Format the filename using both variables
# filename = f"{model_name}_{tuple_str}.pkl.xz"

# with lzma.open('./../measurements/resnet18_32,3,224,224/' + filename, "wb") as file_:
#     pickle.dump(opus_magnum_dict, file_)

# # Load operation dict

# with lzma.open(filename) as file_:
#     saved_dict = pickle.load(file_)

# # print(saved_dict)


# list_attemps = list(saved_dict.items())

# # print(list_attemps[1][1][0])
# # print(list_attemps[1][1][1])
# # print(type(list_attemps[1][1][0]))
