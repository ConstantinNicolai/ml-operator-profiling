import torch
import torch.nn as nn
from collections import defaultdict
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.init as init

# Create a defaultdict to store (layer, input_shape) as keys and counts as values
operation_dict = defaultdict(int)


# def get_parameter_shapes(layer):
#     shapes = {}
#     for name, param in layer.named_parameters():
#         shapes[name] = param.shape
#     if shapes != {}:
#         print("not empty")
#     return shapes


def get_parameter_shapes(layer):
    shapes = {}
    for name, param in layer.named_parameters():
        shapes[name] = param.shape
    return shapes


def dict_to_tuple(d):
    # Convert dictionary to a tuple of sorted key-value pairs
    return tuple(sorted(d.items()))

def check_key_exists(d, input_shape, shapes_dict):
    # Convert the shapes dict to a tuple of sorted key-value pairs
    shapes_tuple = dict_to_tuple(shapes_dict)
    
    for key in d:
        module, existing_input_shape, existing_shapes_tuple = key
        
        # Check if input_shape and shapes match the existing key components
        if existing_input_shape == input_shape and existing_shapes_tuple == shapes_tuple:
            return True
    return False

def check_empty_key_exists(d, input_shape, shapes_dict):
   
    for key in d:
        existing_module, existing_input_shape, existing_shapes_tuple = key
        
        # Check if input_shape and shapes match the existing key components
        if existing_module == module and existing_input_shape == input_shape:
            return True
    return False


def get_repr_module(d, input_shape, shapes_dict):
    # Convert the shapes dict to a tuple of sorted key-value pairs
    shapes_tuple = dict_to_tuple(shapes_dict)
    
    for key in d:
        module, existing_input_shape, existing_shapes_tuple = key
        
        # Check if input_shape and shapes match the existing key components
        if existing_input_shape == input_shape and existing_shapes_tuple == shapes_tuple:
            return module


def get_empty_repr_module(d, input_shape, shapes_dict):

    for key in d:
        existing_module, existing_input_shape, existing_shapes_tuple = key
        
        # Check if input_shape and shapes match the existing key components
        if existing_module == module and existing_input_shape == input_shape:
            return existing_module


# Define the forward hook function
def forward_hook(module, input, output):
    # Check if the layer is a leaf (i.e., it has no children)
    if not len(list(module.children())):
        # Extract the input shape (assuming input is a tuple)
        input_shape = tuple(input[0].size())

        shapes = get_parameter_shapes(module)

        if shapes != {}:
            # print("not empty")
            if check_key_exists(operation_dict, input_shape, shapes):
                # print("I already exist")
                module = get_repr_module(operation_dict, input_shape, shapes)
        # elif shapes == {}:
        #     if check_empty_key_exists(operation_dict, input_shape, shapes):
        #         module = get_empty_repr_module(operation_dict, input_shape, shapes)

             
        
        # Create the key as a tuple of (layer object, input shape)
        key = (module, input_shape, dict_to_tuple(shapes))
        
        # Increment the count for this specific (layer, input_shape) combination
        operation_dict[key] += 1

# Load the ResNet34 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

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

# for key, value in operation_dict.items():
#     print(f'{key}: {value}')


# List all the keys in the defaultdict
lititi = list(operation_dict.keys())


# print(lititi[11])
# print(operation_dict[lititi[11]])
print(lititi[12])
print(operation_dict[lititi[12]])
# print(lititi[13])
# print(operation_dict[lititi[13]])
print(lititi[14])
print(operation_dict[lititi[14]])


print(lititi[14][1] == lititi[12][1])

# print(dir(lititi[14][0]))

print(lititi[14][0]._get_name)
print(type(lititi[14][0]._get_name))
print(id(lititi[14][0]._get_name))
print(id(lititi[12][0]._get_name))
print(lititi[12][0]._get_name)

print(lititi[14][0]._get_name == lititi[12][0]._get_name)