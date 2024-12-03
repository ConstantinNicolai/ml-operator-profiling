import torch
import os
import yaml
import lzma
import pickle
import math

# Load the saved .pt file
dataset_a30notc = torch.load('datasets_fullmodel/dataset_history_A30_no_tc/dataset_20241118_135227.pt', map_location=torch.device('cpu'))
dataset_2080ti = torch.load('datasets_fullmodel/dataset_history_RTX2080TI/dataset_20241118_153430.pt', map_location=torch.device('cpu'))

# print("#########################################")

# print("type of dataset", type(dataset))
# print("tye of dataset entries", type(dataset[0]))




dataset_list_a30notc = [list(item) for item in dataset_a30notc]

dataset_list_2080ti = [list(item) for item in dataset_2080ti]

for i in range(len(dataset_list_a30notc)):
    print("a20notc", dataset_list_a30notc[i][0].__class__.__name__, dataset_list_a30notc[i][1:-1])
    print("2080ti", dataset_list_2080ti[i][0].__class__.__name__, dataset_list_2080ti[i][1:-1])