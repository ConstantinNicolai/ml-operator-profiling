import torch
import torch.nn as nn
from collections import defaultdict
from torchvision.models import resnet18, ResNet18_Weights

# Create a defaultdict to store the counts of (layer, input_shape) pairs
operation_dict = defaultdict(int)

# Define the forward hook function
def forward_hook(module, input, output):
    # Check if the layer is a leaf (i.e., it has no children)
    if not len(list(module.children())):
        # Extract the input shape (assuming input is a tuple)
        input_shape = tuple(input[0].size())
        
        # Use the (module, input_shape) pair as the key in operation_dict
        operation_dict[(module.__class__.__name__, input_shape)] += 1

# Load the ResNet34 model
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Define the input size
input_size = (3, 56, 56)

# Generate random input data
input_data = torch.randn(32, *input_size)

# Register the forward hook only for leaf nodes
for module in model.modules():
    if len(list(module.children())) == 0:  # Register only for leaf nodes (no children)
        module.register_forward_hook(forward_hook)

# Run a forward pass to trigger the hooks
output = model(input_data)

# Print the results
print("Count of similar computational units (layer and input shape):")
for (layer, input_shape), count in operation_dict.items():
    print(f"{layer} with input shape {input_shape}: {count} times")

# Calculate the total count of all computational units
total_count = sum(operation_dict.values())

# Print the total count
print(f"Total count of computational units: {total_count}")


