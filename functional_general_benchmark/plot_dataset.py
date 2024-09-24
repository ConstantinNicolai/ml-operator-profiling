import torch
import os
import yaml
import lzma
import pickle
import math

# Load the saved .pt file
dataset = torch.load('dataset_history/dataset_20240923_133127.pt', map_location=torch.device('cpu'))


dataset_list = [list(item) for item in dataset]

#if we want to be able to plot anything right here, we need to lower the problems dimensionality


print(dir(dataset_list[345][0]))
print(type(dataset_list[345][0].out_channels))

conv2d_list = []
linear_list = []




for item in dataset_list:
    if item[0]._get_name() == "Conv2d":
        conv2d_list.append(item)
    elif item[0]._get_name() == "Linear":
        linear_list.append(item)
        # print("MACs = ",item[0].out_channels*item[1][1]*item[1][2]*item[1][3]*item[0].kernel_size[0]*item[0].kernel_size[1])
        # print(item[0].kernel_size[0],item[0].kernel_size[1])
        # print(" yes, yes , yes !")
    # print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
    # print(item)


for item in conv2d_list:
    print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
    print("MACs = ",item[0].out_channels*item[1][1]*item[1][2]*item[1][3]*item[0].kernel_size[0]*item[0].kernel_size[1])

for item in linear_list:
    print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
    # print(item[0].bias is not None)
    # print(dir(item[0]))
    if item[0].bias is not None:
        print("MACs = ", item[0].out_features*item[0].in_features+item[0].out_features)
    else:
        print("MACs = ", item[0].out_features*item[0].in_features)