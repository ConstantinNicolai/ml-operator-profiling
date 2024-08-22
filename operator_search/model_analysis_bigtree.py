import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision import transforms
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter

# Load the pretrained ResNet-18 model
model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

input_size = (3, 56, 56)


input_data = torch.randn(1, *input_size)


torchinfo_writer = TorchinfoWriter(model,
                                    input_data=input_data,
                                    verbose=0)

torchinfo_writer.construct_model_tree()

torchinfo_writer.show_model_tree(attr_list=['Type', 'Kernel Size', 'Input Size', 'Output Size'])


#WE NEED TO HAEV MORE ATTRIBUTES FOR BOIGTREE SO IT DOES NOT COUTN DIFFERETN LAYERS AS THE SAME BASED ON INPUT AND OUTPUT SIZES