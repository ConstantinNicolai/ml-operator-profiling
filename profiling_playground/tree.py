import torch


import sys


# Check if a command-line argument was provided
if len(sys.argv) != 3:
    print("Usage: python example.py <filepath>")
    sys.exit(1)

# Read the command-line argument
model = sys.argv[1]
weights_par = sys.argv[2]


import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import model, weights_par
from torchvision import transforms
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter

# Load the pretrained ResNet-50 model
model = model(weights=weights_par_Weights.DEFAULT)

input_size = (3, 56, 56)


input_data = torch.randn(1, *input_size)


torchinfo_writer = TorchinfoWriter(model,
                                    input_data=input_data,
                                    verbose=0)

torchinfo_writer.construct_model_tree()

torchinfo_writer.show_model_tree(attr_list=['Input Size', 'Output Size'])