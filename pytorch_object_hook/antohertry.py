import torch
import torch.nn as nn
from collections import defaultdict
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
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
input_size = (3, 8, 8)

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

# print(dir(lititi[14][0]))

print(lititi[12][0].named_modules())

print(lititi[12][0].extra_repr())

print(id(lititi[14][0]._get_name()))
print(id(lititi[12][0]._get_name()))


testconv0 = nn.Conv2d(3, 32, 3, 3, (1,1), bias=False)

testconv1 = nn.Conv2d(3, 32, 3, 3, (1,1), bias=False)

testattent = nn.MultiheadAttention(6, 3, dropout=0, bias=False)


print(id(testconv0._get_name()))
print(id(testconv1._get_name()))

print(testconv0._get_name() == testconv1._get_name())

print(testattent._get_name())
print(testattent.extra_repr())

print(type(lititi[12][0].extra_repr()))


# for i in range(len(lititi)):
#     print(lititi[i][0].extra_repr())


# # Create a defaultdict to store (layer, input_shape) as keys and counts as values
# opus_magnum_dict = defaultdict(int)





# def get_equivalent_layer(d, input_shape):

#     for key in d:
#         existing_module, existing_input_shape = key

#         if existing_module._get_name() == module._get_name() and existing_module.extra_repr() == module.extra_repr() and existing_input_shape == input_shape:
#             return existing_module ,existing_input_shape


# # Define the forward hook function
# def forward_hook_new(module, input, output):

#     # print("################")

#     # for key, value in opus_magnum_dict.items():
#     #     print(f'{key}: {value}')

#     # print("################")

#     # print(module._get_name())
#     # Check if the layer is a leaf (i.e., it has no children)
#     if not len(list(module.children())):
#         # Extract the input shape (assuming input is a tuple)
#         input_shape = tuple(input[0].size())

#         # if check_equivalent_layer_exists(opus_magnum_dict, input_shape):
#         #     # print("I already exist")
#         #     module, input_shape = get_equivalent_layer(opus_magnum_dict, input_shape)


#         # Create the key as a tuple of (layer object, input shape)
#         key = (module, input_shape)
        
#         # Increment the count for this specific (layer, input_shape) combination
#         opus_magnum_dict[key] += 1












# print(model)

# # Register the forward hook only for leaf nodes
# for module in model.modules():
#     module.register_forward_hook(forward_hook_new)


# # Run a forward pass to trigger the hooks
# output = model(input_data)


# for key, value in opus_magnum_dict.items():
#     print(f'{key}: {value}')


# newdict = defaultdict(int)


# for key in opus_magnum_dict:
#     log_module, log_inshape = key

#     for key in newdict:
#         new_module, new_inshape = key
#         if new_module == log_module:
#             print("hit")
#             fin_mod = new_module
#             fin_inshape = new_inshape
        
#     fin_mod = log_module
#     fin_inshape = log_inshape

#     fin_key = fin_mod, fin_inshape
#     newdict[fin_key] += 1








# # Define the forward hook function
# def forward_hook_new(module, input, output):

#     # Check if the layer is a leaf (i.e., it has no children)
#     if not len(list(module.children())):
#         # Extract the input shape (assuming input is a tuple)
#         input_shape = tuple(input[0].size())

#         # Create the key as a tuple of (layer object, input shape)
#         key = (module, input_shape)
        
#         # Increment the count for this specific (layer, input_shape) combination
#         opus_magnum_dict[key] += 1





# Create the defaultdict to count layer-input occurrences
opus_magnum_dict = defaultdict(int)  # {key: count}

# Define the forward hook function
def forward_hook_new(module, input, output):
    # Check if the layer is a Conv2d and has no children
    if isinstance(module, torch.nn.Conv2d) and not len(list(module.children())):
        # Extract the input shape (assuming input is a tuple)
        input_shape = tuple(input[0].size())

        # Create a key based on module name, extra_repr, and input shape
        key = (module._get_name(), module.extra_repr(), input_shape)

        # Increment the count for this specific configuration
        opus_magnum_dict[key] += 1






class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        # Define two Conv2d layers with similar structure
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Define ReLU activations
        self.relu = nn.ReLU()
        
        # A fully connected layer
        # You need to know the input size to the linear layer after convolutions
        self.fc = nn.Linear(16 * 8 * 8, 10)  # Assuming the input image is 8x8 after convolutions

    def forward(self, x):
        # Forward pass through the Conv2d layers and ReLU activations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        # Now flatten the output to fit into the fully connected layer
        # x.size(0) is the batch size, so we reshape the tensor to [batch_size, -1]
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Pass it through the fully connected layer
        x = self.fc(x)
        
        return x

# Example of creating and printing the model
model = TestModel()


# Register the forward hook only for leaf nodes
for module in model.modules():
    module.register_forward_hook(forward_hook_new)


output = model(input_data)


for key, value in opus_magnum_dict.items():
    print(f'{key}: {value}')


model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Register the forward hook only for leaf nodes
for module in model.modules():
    # if len(list(module.children())) == 0:
    module.register_forward_hook(forward_hook_new)


opus_magnum_dict = defaultdict(int)  # {key: count}

output = model(input_data)

# Function to sum counts for Conv2d layers in the dictionary
def sum_conv2d_counts(opus_magnum_dict):
    return sum(opus_magnum_dict.values())
# Example of summing counts for Conv2d layers
total_conv2d_count = sum_conv2d_counts(opus_magnum_dict)
# print(f"Total count of Conv2d layer-input combinations: {total_conv2d_count}")

# for key, value in opus_magnum_dict.items():
#     print(f'{key}: {value}')





# def check_equivalent_layer_exists(d, input_shape):


#     for key in d:
#         ex_module, ex_module_name, ex_module_extra, ex_input_shape = key

#         if ex_module_name == module._get_name() and ex_module_extra == module.extra_repr() and ex_input_shape == input_shape:
#             return True
    
#     return False










# # Define the forward hook function
# def forward_hook_new(module, input, output):
#     # Check if the layer is a Conv2d and has no children
#     if isinstance(module, torch.nn.Conv2d) and not len(list(module.children())):
#         # Extract the input shape (assuming input is a tuple)
#         input_shape = tuple(input[0].size())

#         if check_equivalent_layer_exists(current_dict, input_shape):
#             print(double_layer)

#         # Create a key based on module name, extra_repr, and input shape
#         key = (module, module._get_name(), module.extra_repr(), input_shape)

#         # Increment the count for this specific configuration
#         current_dict[key] += 1


# model = resnet18(weights=ResNet18_Weights.DEFAULT)

# # Register the forward hook only for leaf nodes
# for module in model.modules():
#     # if len(list(module.children())) == 0:
#     module.register_forward_hook(forward_hook_new)


# current_dict = defaultdict(int)  # {key: count}

# output = model(input_data)


# for key, value in current_dict.items():
#     print(f'{key}: {value}')





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


for key, value in opus_magnum_dict.items():
    print(f'{key}: {value}')