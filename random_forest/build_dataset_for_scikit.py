import torch
import os
import yaml
import lzma
import pickle
import math
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def add_elements_based_on_B(outer_list, B, transform_func):
    """
    Adds elements to the inner lists of outer_list based on attributes of the corresponding
    elements in list B.

    Args:
        outer_list (list of lists): The main list of lists to be modified.
        B (list): A list whose elements correspond to each inner list in outer_list.
        transform_func (function): A function that takes an element of B and returns
                                   the value(s) to add to the corresponding inner list.

    Returns:
        list: The updated outer_list with modified inner lists.
    """
    return [inner_list + transform_func(b_elem) for inner_list, b_elem in zip(outer_list, B)]


# Load the saved .pt file
dataset = torch.load('../functional_general_benchmark/datasets_finalbench/dataset_history_A30/dataset_20241117_202731.pt', map_location=torch.device('cpu'))


dataset_list = [list(item) for item in dataset]


# Create a new list for the random forest inputs
rf_input_list = [
    [x for i, x in enumerate(inner_list) if i in {0,1}]
    for inner_list in dataset_list
]

list_of_pytorch_objects = [row[0] for row in rf_input_list]

print(list_of_pytorch_objects[-1])

# print(list_of_pytorch_objects[0:4])

# for i in list_of_pytorch_objects:
#     try:
#         if type(i.kernel_size) is int:
#             print(i.kernel_size)
#             print(i._get_name())
#     except:
#         continue
categories = [[row[0]._get_name()] for row in rf_input_list] 

# print(categories)

# Initialize and fit the encoder
encoder = OneHotEncoder()
encoded = encoder.fit_transform(categories).toarray()

# Remove the original category column from X and append the encoded values
rf_input_list_encoded = [list(encoded[i]) + row[1:] for i, row in enumerate(rf_input_list)]

# rf_input_list_no_tuple = [inner_list[:-1] + list(inner_list[-1]) for inner_list in rf_input_list_encoded]


rf_input_list_no_tuple = [
    inner_list[:-1] + list(inner_list[-1])[:4] + [-1] * (4 - len(list(inner_list[-1])[:4]))
    for inner_list in rf_input_list_encoded
]


# print(rf_input_list_no_tuple[0:4])

# print(len(encoded[0]))

def transform_func(torchobj):
    try:
        # Attempt to access the 'bias' attribute
        bias = torchobj.bias
        if bias is None:
            return [0,1]
        else:
            return [1,1]
    except AttributeError:
        # Handle cases where 'bias' does not exist
        return [-1,0]

updated_list = add_elements_based_on_B(rf_input_list_no_tuple, list_of_pytorch_objects, transform_func)

print(updated_list[-1])
print(updated_list[-2])
# for row in updated_list:
#     print(len(row))

def transform_func(torchobj):
    try:
        return [torchobj.in_channels,1]
    except AttributeError:
        # Handle cases where 'bias' does not exist
        return [-1,0]

updated_list = add_elements_based_on_B(updated_list, list_of_pytorch_objects, transform_func)

def transform_func(torchobj):
    try:
        return [torchobj.out_channels,1]
    except AttributeError:
        # Handle cases where 'bias' does not exist
        return [-1,0]
        
updated_list = add_elements_based_on_B(updated_list, list_of_pytorch_objects, transform_func)



def transform_func(torchobj):
    try:
        if type(torchobj.kernel_size) is int:
            return [torchobj.kernel_size,torchobj.kernel_size,1]
        else:
            return list(torchobj.kernel_size)+[1]
            print(kuckuck)
    except AttributeError:
        # Handle cases where 'bias' does not exist
        return [-1,-1,0]
        
updated_list = add_elements_based_on_B(updated_list, list_of_pytorch_objects, transform_func)



print(updated_list[0:4])

# Create a new list for the random forest targets
rf_target_list = [
    [x for i, x in enumerate(inner_list) if i in {2,8}]
    for inner_list in dataset_list
]  

print(rf_target_list[55])



# for k in range(len(dataset_list)):
#     if dataset_list[k][0]._get_name() == 'ConvTranspose2d':
#         print(dataset_list[k][0]._get_name())
#         print(dir(dataset_list[k][0]))
#         print(dataset_list[k][0])
#         # print(dataset_list[k][0].parameters)
#         break
        



