import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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

# Create a random input tensor with 3 channels and 32x32 dimensions
input_tensor = torch.randn(1, input_channels, 32, 32).to(device)

# Forward pass through the model
output = model(input_tensor)

# Print the model and the output
print(type(model))

print(model)


# Create a LayerInfo struct to store the second layer, input size, and batch size
layer_info = LayerInfo(
    layer=model.conv2,
    input_size=(16, 20, 20),
    batch_size=32
)

# Print the stored information
print(layer_info)


# Create an input tensor with the specified input size and batch size
input_tensor = torch.randn(layer_info.batch_size, layer_info.input_size[0], layer_info.input_size[1], layer_info.input_size[2])


# Run the stored layer with the input tensor
output = layer_info.layer(input_tensor)



