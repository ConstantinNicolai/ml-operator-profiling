import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VisionTransformer
from torchvision import transforms
import re


# Load the pretrained ResNet-18 model
model = VisionTransformer

print(model)
# encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
# model = nn.TransformerEncoder(encoder_layer, num_layers=6)


# # Initialize a list to store the layer information
# layer_info_list = []

# # Iterate through the model's layers
# for name, layer in model.named_modules():
#     if len(list(layer.children())) == 0:  # Focus on individual layers, skip container layers
#         # Get the string representation of the layer
#         layer_repr = repr(layer)
#         # Split by ':' and take everything after the first occurrence
#         layer_details = layer_repr.split(':', 1)[-1].strip()
#         # Extract the function type and parameters
#         layer_type = layer.__class__.__name__
#         layer_parameters = layer_details[len(layer_type)+1:-1]  # Remove the layer type and parentheses

#         # Append the information to the list
#         layer_info_list.append({
#             "Type": layer_type,
#             "Parameters": layer_parameters
#         })

# # Custom function to split parameters while respecting brackets
# def split_params(param_str):
#     # Regular expression to match parameters, respecting nested brackets
#     matches = re.findall(r'(\w+=[^,()]+(?:\([^)]*\))?|[^,()]+(?:\([^)]*\))?)', param_str)
#     return matches

# # Convert the list of layer information to a pandas DataFrame
# df = pd.DataFrame(layer_info_list)

# # Apply the custom split function to the Parameters column
# params_df = df['Parameters'].apply(split_params).apply(pd.Series)

# # Combine the split parameters with the original DataFrame
# df = pd.concat([df.drop(columns=['Parameters']), params_df], axis=1)


# # Filter the DataFrame to keep only rows with 'Conv2d' or 'Linear' in the 'Type' column
# filtered_df = df[df['Type'].isin(['Conv2d', 'Linear'])]

# # Display the filtered DataFrame
# # print(df)

# def my_function(x):
#     return x + 1

# # Assign the function to a variable
# peter = my_function

# # # Get the function's name as a string
# # function_name = model.__name__

# # Get the class name
# model_class_name = type(model).__name__

# print(model_class_name)  # Output: my_function