import torch
import torch.nn as nn
from collections import defaultdict
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.init as init
import pickle
import lzma


# Define the input size
input_size = (3, 8, 8)

# Generate random input data
input_data = torch.randn(32, *input_size)


# Define the forward hook function
def forward_hook_new(module, input, output):
    # Check if the layer is a Conv2d and has no children
    if not len(list(module.children())):
        # Extract the input shape (assuming input is a tuple)
        input_shape = tuple(input[0].size())

        # Create a key based on module name, extra_repr, and input shape
        key = (module._get_name(), module.extra_repr(), input_shape)

        # Check if the key exists in the dict
        if opus_magnum_dict[key][0] is None:
            # If it's the first occurrence, store the module object and set the count to 1
            opus_magnum_dict[key] = [module, 1]
        else:
            # If we've seen this combination before, increment the count
            opus_magnum_dict[key][1] += 1


model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Register the forward hook only for leaf nodes
for module in model.modules():
    # if len(list(module.children())) == 0:
    module.register_forward_hook(forward_hook_new)


# Create the defaultdict to store the module (first occurrence) and the count
opus_magnum_dict = defaultdict(lambda: [None, 0])  # {key: [first_module_object, count]}


output = model(input_data)


opus_magnum_dict = dict(opus_magnum_dict)

# for key, value in opus_magnum_dict.items():
#     print(f'{key}: {value}')



with lzma.open("ops.pkl.xz", "wb") as file_:
    pickle.dump(opus_magnum_dict, file_)

# Load operation dict

with lzma.open("ops.pkl.xz") as file_:
    saved_dict = pickle.load(file_)

# print(saved_dict)


list_attemps = list(saved_dict.items())

print(list_attemps[1][1][0])
print(list_attemps[1][1][1])
print(type(list_attemps[1][1][0]))
