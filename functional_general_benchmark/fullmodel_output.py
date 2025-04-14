import torch
import os
import yaml
import lzma
import pickle
import math
import argparse
import re


# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="set the path for the dataset")
parser.add_argument("--path", type=str, required=True, help="Specify the path for the dataset")


# Parse arguments
args = parser.parse_args()
path = args.path

# Load the saved .pt file

# This sets the dataset of operations used to sum up from

string = "../functional_general_benchmark/datasets_fullmodel_inf_validation/dataset_history_" + path

# Regular expression to match filenames like dataset_YYYYMMDD_HHMMSS.pt
pattern = re.compile(r"dataset_(\d{8})_(\d{6})\.pt")

# Get all matching files
files = [
    f for f in os.listdir(string)
    if pattern.match(f)
]

# Extract the latest file based on timestamp
if files:
    latest_file = max(files, key=lambda f: pattern.match(f).groups())
    latest_file_path = os.path.join(string, latest_file)

    # Load the dataset
    dataset = torch.load(latest_file_path, map_location=torch.device("cpu"))


# Load the saved .pt file
#dataset = torch.load('datasets_fullmodel_validation/dataset_history_A30/dataset_20250303_181128.pt', map_location=torch.device('cpu'))


dataset_list = [list(item) for item in dataset]

# for item in dataset_list:
#     print(item[16], item[1])
#     a = item[2]
#     print(f"{a:.50f}".rstrip('0').rstrip('.'))

#     a = item[3]
#     print(f"{a:.50f}".rstrip('0').rstrip('.'))





for item in dataset_list:
    cucu = item[1]
    square_brackets = [cucu[0],cucu[1],cucu[2],cucu[3]]
    print(item[16], square_brackets)
    a = item[2]*1000
    print(f"{a:.50f}".rstrip('0').rstrip('.'))
    # a = item[15]*1000
    # print(f"{a:.50f}".rstrip('0').rstrip('.'))
    a = item[3]
    print(f"{a:.50f}".rstrip('0').rstrip('.'))
    # a = item[5]
    # print(f"{a:.50f}".rstrip('0').rstrip('.'))

