import torch
import torch.nn as nn
from collections import defaultdict
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.init as init
import pickle
import lzma


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



with lzma.open("ops.pkl.xz") as file_:
    saved_dict = pickle.load(file_)

# print(saved_dict)


list_attemps = list(saved_dict.items())

# print(list_attemps[1][1][0])
# print(list_attemps[1][1][1])
# print(type(list_attemps[1][1][0]))


for i in range(len(list_attemps)):
    print(list_attemps[i][1][1])
    print(list_attemps[i][1][0])
    print(list_attemps[i][0][2])