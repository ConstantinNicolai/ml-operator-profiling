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

print(model)