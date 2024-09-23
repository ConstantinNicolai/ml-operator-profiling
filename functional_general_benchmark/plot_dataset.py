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


for item in dataset_list:
    print(item[0]._get_name(), item[0].extra_repr(), item[1])
    # print(item)