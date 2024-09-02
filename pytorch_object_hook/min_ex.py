import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import maxvit_t, MaxVit_T_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from dataclasses import dataclass
import numpy as np


@dataclass
class LayerInfo:
    layer: nn.Module
    input_size: tuple
    batch_size: int



# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple neural network with two convolutional layers
class TwoConvLayerNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(TwoConvLayerNet, self).__init__()
        # First convolutional layer: (input_channels, 16 filters, 3x3 kernel)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: (16 filters, 32 filters, 3x3 kernel)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Fully connected layer to map from conv features to the output classes
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU and max pooling
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        
        # Apply the second convolutional layer followed by ReLU and max pooling
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output from the conv layers
        x = x.view(x.size(0), -1)
        
        # Apply the fully connected layer
        x = self.fc(x)
        return x

# Example usage
input_channels = 3  # Number of input channels (e.g., 3 for RGB images)
num_classes = 10    # Number of output classes

# Instantiate the model
model = TwoConvLayerNet(input_channels, num_classes).to(device)
model = resnet34(weights=ResNet34_Weights.DEFAULT).to(device)

# # Create a random input tensor with 3 channels and 32x32 dimensions
# input_tensor = torch.randn(1, input_channels, 32, 32).to(device)

# # Forward pass through the model
# output = model(input_tensor)

# Print the model and the output
# print(type(model))

# print(model)


# Create a LayerInfo struct to store the second layer, input size, and batch size
layer_info = LayerInfo(
    layer=model.layer1[2],
    input_size=(16, 20, 20),
    batch_size=32
)

# Print the stored information
# print(layer_info)


# # Create an input tensor with the specified input size and batch size
# input_tensor = torch.randn(layer_info.batch_size, layer_info.input_size[0], layer_info.input_size[1], layer_info.input_size[2])


# # Run the stored layer with the input tensor
# output = layer_info.layer(input_tensor)



# print(dir(model.layer1))

# print(model.layer1.children)

child = list(model.layer1.children())[0]  # Get the first child
# print(child)

# print(dir(child))

# print(child.named_children)

subchild = list(child.named_children())[0]

print(subchild)

print(type(subchild[1]))

layer_info1 = LayerInfo(
    layer=subchild[1],
    input_size=(16, 20, 20),
    batch_size=32
)

print(layer_info1)


print(list(list(model.layer1.children())[0].named_children())[0][1])


print('############################################################')


print(dir(list(model.named_children())[0]))

# depending on the model architecture there wil be a lot to iterate over

# in this case we need the direct sublayers of model, and then the names layers like layer1
# their children, which have named_children, and those are tuples of the name and the pytorch object

print('############################################################')

model = maxvit_t(weights=MaxVit_T_Weights.DEFAULT).to(device)


# print(dir(model))

# maxvir = str(dir(model))

# print(dir(list(model.named_children())[0][1]))

# print(list(model.children())[0])

# print(dir(list(list(model.children())[0].children())[0]))

# print(list(model.children())[1][2][1])

# print(list(list(model.children())[1].children())[2])


# for i in range(len(list(model.children()))):
#     print(i)
#     for j in range(len(list(list(model.children())[i].children()))):
#         print(j)
#         for k in range(len())



gustave = list(model.children())

# print(gustave[1][0])


# print(test)

# print(gustave)

print(len(gustave))


tt= 0
for i in range(len(gustave)):
    ttnew = len(gustave[i])
    if ttnew > tt:
        tt = ttnew

print(tt)

tb= 0
for i in range(len(gustave)):
    for k in range(tt):
        try:
            tbnew = len(gustave[i][k])
            if tbnew > tb:
                tb = tbnew
        except Exception as e:
            continue

print(tb)



tc= 0
for i in range(len(gustave)):
    for k in range(tt):
        for j in range(tb):
            try:
                tcnew = len(gustave[i][k][j])
                if tcnew > tc:
                    tc = tbnew
            except Exception as e:
                continue


print(tc)

# for i in range(len(gustave)):
#     for k in range(tt):
#         try:
#             for j in range(tb):
#                 try:
#                     print(gustave[i][k][j])
#                 except Exception as e:
#                     continue
#         except Exception as e:
#             continue



# print(len(gustave)[:][:])

# for i in range(4):
#     print('t000est')

print('############################################################')

model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)

print(model)

# print(dir(model))

# inception = str(dir(model))

# assert inception == inception, "they're not the same"


