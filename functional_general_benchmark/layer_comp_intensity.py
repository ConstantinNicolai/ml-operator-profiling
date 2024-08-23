import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter
import pandas

# Load the pretrained ResNet-18 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
model = nn.TransformerEncoderLayer(d_model=512, nhead=8)

# model = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = torch.rand(10, 32, 512)

input_size = (3, 224, 224)


input_data = torch.randn(1, *input_size)

input_data = torch.rand(10, 32, 512)


torchinfo_writer = TorchinfoWriter(model,
                                    input_data=input_data,
                                    verbose=0)

torchinfo_writer.construct_model_tree()

df = torchinfo_writer.get_dataframe()

print(df)

print(model)

#torchinfo_writer.show_model_tree(attr_list=['Parameters'])

