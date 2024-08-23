import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision import transforms
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter
import torchinfo
import re


# Load the pretrained ResNet-18 model
model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)


# Initialize a list to store the layer information
layer_info_list = []

# Iterate through the model's layers
for name, layer in model.named_modules():
    if len(list(layer.children())) == 0:  # Focus on individual layers, skip container layers
        # Get the string representation of the layer
        layer_repr = repr(layer)
        # Split by ':' and take everything after the first occurrence
        layer_details = layer_repr.split(':', 1)[-1].strip()
        # Extract the function type and parameters
        layer_type = layer.__class__.__name__
        layer_parameters = layer_details[len(layer_type)+1:-1]  # Remove the layer type and parentheses

        # Append the information to the list
        layer_info_list.append({
            "Type": layer_type,
            "Parameters": layer_parameters
        })

# Custom function to split parameters while respecting brackets
def split_params(param_str):
    # Regular expression to match parameters, respecting nested brackets
    matches = re.findall(r'(\w+=[^,()]+(?:\([^)]*\))?|[^,()]+(?:\([^)]*\))?)', param_str)
    return matches

# Convert the list of layer information to a pandas DataFrame
df = pd.DataFrame(layer_info_list)

# Apply the custom split function to the Parameters column
params_df = df['Parameters'].apply(split_params).apply(pd.Series)

# Combine the split parameters with the original DataFrame
df = pd.concat([df.drop(columns=['Parameters']), params_df], axis=1)

# Display the DataFrame
print(df)

df.to_csv("dynamic_model_layers.csv", index=False)





# # Initialize a list to store the layer information
# layer_info_list = []

# # Iterate through the model's layers
# for name, layer in model.named_modules():
#     # Skip the parent module itself and focus on individual layers
#     if len(list(layer.children())) == 0:
#         layer_info = {"Layer Name": name, "Type": layer.__class__.__name__}
        
#         # Get all attributes of the layer
#         for attribute_name, attribute_value in vars(layer).items():
#             # Convert attribute values to a string representation if they are not basic types
#             if not isinstance(attribute_value, (int, float, str, tuple)):
#                 attribute_value = str(attribute_value)
#             layer_info[attribute_name] = attribute_value
        
#         # Add the number of parameters
#         layer_info["Number of Parameters"] = sum(p.numel() for p in layer.parameters())
        
#         # Add whether the parameters are trainable
#         layer_info["Trainable"] = any(p.requires_grad for p in layer.parameters())
        
#         # Append to the list
#         layer_info_list.append(layer_info)

# # Convert the list of layer information to a pandas DataFrame
# df = pd.DataFrame(layer_info_list)

# # Display the DataFrame
# print(df)

# df.to_csv("dynamic_model_layers.csv", index=False)





# # Use torchinfo to get a detailed summary
# summary = torchinfo.summary(model, input_size=(1, 3, 224, 224), col_names=("output_size", "num_params"))

# # Convert summary to a list of dicts
# layer_info_list = []
# for layer in summary.summary_list:
#     layer_info = {
#         "Layer Name": layer.module_name,
#         "Output Shape": layer.output_size,
#         "Number of Parameters": layer.num_params,
#         "Trainable": layer.trainable,
#     }
#     layer_info_list.append(layer_info)

# # Convert list of dicts to a pandas DataFrame
# df = pd.DataFrame(layer_info_list)

# # Display the DataFrame
# print(df)



# # Extract layer details
# def extract_layer_details(model):
#     layer_details = []
#     for name, layer in model.named_children():
#         if isinstance(layer, nn.ModuleList) or isinstance(layer, nn.Sequential):
#             for sub_name, sub_layer in layer.named_children():
#                 layer_details.append({
#                     'Layer Name': f'{name}.{sub_name}',
#                     'Layer Type': str(sub_layer.__class__.__name__),
#                     'Input Shape': 'N/A',  # Input shapes are not straightforward to extract
#                     'Output Shape': 'N/A'  # Output shapes are not straightforward to extract
#                 })
#         else:
#             layer_details.append({
#                 'Layer Name': name,
#                 'Layer Type': str(layer.__class__.__name__),
#                 'Input Shape': 'N/A',
#                 'Output Shape': 'N/A'
#             })
#     return layer_details

# # Create DataFrame
# layer_details = extract_layer_details(model)
# df = pd.DataFrame(layer_details)

# # Print DataFrame
# print(df)
# print(model)