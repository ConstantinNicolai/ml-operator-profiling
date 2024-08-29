#### Go here to find model and weights name and capitalization https://pytorch.org/vision/stable/models.html#classification






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






from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model


# Example usage:
df_counts = process_model(input_size=(3, 224, 224), filter_types=['Conv2d', 'Linear'], exclude_string='downsample')

print(df_counts)




# def convert_digit_strings(my_list):
#     # Convert strings of digits to integers
#     converted_list = [
#         int(item.strip()) if item.strip().isdigit() else item
#         for item in my_list
#     ]
#     return converted_list


# def parse_param_string(param_string):
#     """ Convert a parameter string into a dictionary format """
#     try:
#         # Convert the string into a dictionary-like string and parse it
#         param_dict = ast.literal_eval('{' + param_string + '}')
#         return param_dict
#     except (SyntaxError, ValueError):
#         # Handle errors in case the string cannot be parsed
#         return None

# def process_list(data_list):
#     """ Process a list to convert parameter strings to dictionaries """
#     processed_list = []
    
#     for item in data_list:
#         if isinstance(item, str):
#             # Identify if the string is a parameter string
#             parsed_item = parse_param_string(item)
#             if parsed_item is not None:
#                 processed_list.append(parsed_item)
#             else:
#                 processed_list.append(item)
#         else:
#             # If it's not a string, keep it unchanged
#             processed_list.append(item)
    
#     return processed_list


import ast

def parse_value(value):
    # Strip leading and trailing whitespace
    value = value.strip()
    
    # Handle boolean values
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    
    # Handle integers
    try:
        return int(value)
    except ValueError:
        pass
    
    # Handle tuples, e.g., "kernel_size=(3, 3)"
    if value.startswith('(') and value.endswith(')'):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    
    # Return value as string if no other match
    return value

def convert_list_to_params(param_list):
    result = []
    
    for item in param_list:
        item = item.strip()
        
        # Handle key-value pairs
        if '=' in item:
            key, value = item.split('=', 1)
            result.append((key.strip(), parse_value(value.strip())))
        else:
            # Handle standalone values
            result.append(parse_value(item))
    
    return result












def call_function_from_df(df, index):
    # Check if index is within the DataFrame's range
    if index not in df.index:
        raise IndexError(f"Index {index} is out of bounds.")
    
    # Extract the row at the given index
    row = df.loc[index]
    
    # Get the function name and parameters
    func_name = row['Type']
    print(type(row['input_channels']))
    params = row[['input_channels', 'output_channels', 'kernel_size', 'stride', 'padding', 'bias' ]].tolist()
    params = [item for item in params if not pd.isna(item)]


    print(params)


        # Retrieve the function from the library
    func = getattr(torch.nn, func_name)
    

    print(params)
    print(func)
    # Call the function with parameters
    model = func(256,256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model = func(*params)


    return model

call_function_from_df(df_counts, 2)


print(df_counts.dtypes)