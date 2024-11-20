import torch
import os
import yaml
import lzma
import pickle
import math

# Load the saved .pt file
dataset = torch.load('datasets_fullmodel/dataset_history_RTX2080TI/dataset_20241118_153430.pt', map_location=torch.device('cpu'))

dataset_list = [list(item) for item in dataset]

for item in dataset_list:
    print(item[16], item[1])
    a = item[2]
    print(f"{a:.50f}".rstrip('0').rstrip('.'))
    a = item[15]
    print(f"{a:.50f}".rstrip('0').rstrip('.'))
    a = item[3]
    print(f"{a:.50f}".rstrip('0').rstrip('.'))
    a = item[5]
    print(f"{a:.50f}".rstrip('0').rstrip('.'))

