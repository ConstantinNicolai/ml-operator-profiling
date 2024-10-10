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
import time
import math
import os
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new, process_log_file,get_latest_dataset_file, load_latest_dataset, save_dataset



# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = get_model_and_weights('convnext_base', 'ConvNeXt_Base_Weights')

model = model.to(device)

print('convnext_base', '(32, 3, 384, 384)')

ifmap = torch.randn((32, 3, 384, 384)).cuda()


with torch.no_grad():
    for g in range(2000):
        output = model(ifmap)
        print(g)

print("success")

# with torch.no_grad():
#     output = model(ifmap)
#     print("success")
#     nida = model(ifmap)
#     print("success")