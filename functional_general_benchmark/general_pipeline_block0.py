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
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model



args = parse_model_and_weights()  # Get the parsed arguments

# Access the variables
model_name = args.model
weights_name = args.weights
input_size = args.input_size

model = get_model_and_weights(model_name, weights_name)



##########################################################################


# Generate random input data
input_data = torch.randn(32, *input_size)


# Define the forward hook function
def forward_hook_new(module, input, output):
    # Check if the layer is a Conv2d and has no children
    if not len(list(module.children())):
        # Extract the input shape (assuming input is a tuple)
        input_shape = tuple(input[0].size())

        # Create a key based on module name, extra_repr, and input shape
        key = (module._get_name(), module.extra_repr(), input_shape)

        # Check if the key exists in the dict
        if opus_magnum_dict[key][0] is None:
            # If it's the first occurrence, store the module object and set the count to 1
            opus_magnum_dict[key] = [module, 1]
        else:
            # If we've seen this combination before, increment the count
            opus_magnum_dict[key][1] += 1



# Register the forward hook only for leaf nodes
for module in model.modules():
    # if len(list(module.children())) == 0:
    module.register_forward_hook(forward_hook_new)


# Create the defaultdict to store the module (first occurrence) and the count
opus_magnum_dict = defaultdict(lambda: [None, 0])  # {key: [first_module_object, count]}


output = model(input_data)


opus_magnum_dict = dict(opus_magnum_dict)

# for key, value in opus_magnum_dict.items():
#     print(f'{key}: {value}')


tuple_str = "_".join(map(str, input_size))

# Format the filename using both variables
filename = f"{model_name}_{tuple_str}.pkl.xz"

with lzma.open(filename, "wb") as file_:
    pickle.dump(opus_magnum_dict, file_)

# Load operation dict

with lzma.open(filename) as file_:
    saved_dict = pickle.load(file_)

# print(saved_dict)


list_attemps = list(saved_dict.items())

# print(list_attemps[1][1][0])
# print(list_attemps[1][1][1])
# print(type(list_attemps[1][1][0]))
