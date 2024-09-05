import torch
import torch.nn as nn
from collections import defaultdict
from torchvision.models import resnet18, ResNet18_Weights

# Initialize a dictionary to track the count of each layer based on its parameter shapes and input shape
operation_dict = {}

def get_parameter_shapes(layer):
    """ Get a dictionary of parameter shapes for the given layer """
    shapes = {}
    for name, param in layer.named_parameters():
        shapes[name] = tuple(param.shape)  # Convert shape to tuple for hashability
    return shapes

def forward_hook(module, input, output):
    """ Forward hook function to track layers based on parameter shapes and input shape """
    # Check if the layer is a leaf (i.e., it has no children)
    if not len(list(module.children())):
        # Extract the input shape (assuming input is a tuple)
        input_shape = tuple(input[0].size())
        
        # Get the parameter shapes for the current layer
        param_shapes = get_parameter_shapes(module)
        
        # Create a key for the layer based on its parameter shapes and input shape
        key = (tuple(param_shapes.items()), input_shape)
        
        # Increment the count for this specific (layer, input_shape) combination
        if key in operation_dict:
            operation_dict[key] += 1
        else:
            operation_dict[key] = 1

# Load the ResNet34 model
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Define the input size
input_size = (3, 56, 56)

# Generate random input data
input_data = torch.randn(32, *input_size)


# Register the forward hook for all layers
for layer in model.children():
    layer.register_forward_hook(forward_hook)

# Run a forward pass to trigger the hooks
output = model(input_data)

for key, value in operation_dict.items():
    print(f'{key}: {value}')
