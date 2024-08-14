import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet18_Weights
from torchvision import transforms
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter

# Load the pretrained ResNet-50 model
model = resnet18(weights=resnet18_Weights.DEFAULT)

input_size = (3, 56, 56)


input_data = torch.randn(1, *input_size)


torchinfo_writer = TorchinfoWriter(model,
                                    input_data=input_data,
                                    verbose=0)

torchinfo_writer.construct_model_tree()

torchinfo_writer.show_model_tree(attr_list=['Input Size', 'Output Size'])