import pickle
import lzma
from collections import defaultdict

import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


operation_dict = defaultdict(int)
def get_children(layer: nn.Module, args) -> None:

    if not len(list(layer.children())):
        operation_dict[layer] += 1





# Define the input size
input_size = (3, 56, 56)

# Generate random input data and move it to the GPU
input_data = torch.randn(32, *input_size)


model = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
# model = resnet34(weights=ResNet34_Weights.DEFAULT)


get_children(model, input_data)

print(operation_dict)


# Store operation dict

with lzma.open("ops.pkl.xz", "wb") as file_:
    pickle.dump(operation_dict, file_)

# Load operation dict

with lzma.open("ops.pkl.xz") as file_:
    operation_dict = pickle.load(file_)

print(operation_dict)