import argparse
import importlib
import torch
from torchvision import models, transforms
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import re
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter


def get_model_and_weights(model_name, weights_name):
    # Load the model function from torchvision.models
    model_module = importlib.import_module('torchvision.models')
    model_fn = getattr(model_module, model_name, None)
    if model_fn is None:
        raise ValueError(f"Model '{model_name}' is not available in torchvision.models.")
    
    # Load the weights class from torchvision.models
    weights_class = None
    try:
        weights_module = importlib.import_module('torchvision.models')
        weights_class = getattr(weights_module, weights_name, None)
    except ImportError:
        pass
    
    # If weights class not found in torchvision.models, check torchvision.transforms
    if weights_class is None:
        try:
            weights_module = importlib.import_module('torchvision.transforms')
            weights_class = getattr(weights_module, weights_name, None)
        except ImportError:
            pass
    
    if weights_class is None:
        raise ValueError(f"Weights class '{weights_name}' is not available in torchvision.models or torchvision.transforms.")
    
    # Load the weights
    weights_fn = weights_class.DEFAULT
    model = model_fn(weights=weights_fn)
    return model


# def extract_layer_info(model):
#     """
#     Extracts layer information from the given PyTorch model and returns it as a pandas DataFrame.
    
#     Args:
#         model: A PyTorch model (e.g., torchvision.models.ResNet50)
    
#     Returns:
#         pd.DataFrame: A DataFrame containing the type and parameters of each layer.
#     """

#     # Initialize a list to store the layer information
#     layer_info_list = []

#     # Iterate through the model's layers
#     for name, layer in model.named_modules():
#         if len(list(layer.children())) == 0:  # Focus on individual layers, skip container layers
#             # Get the string representation of the layer
#             layer_repr = repr(layer)
#             # Split by ':' and take everything after the first occurrence
#             layer_details = layer_repr.split(':', 1)[-1].strip()
#             # Extract the function type and parameters
#             layer_type = layer.__class__.__name__
#             layer_parameters = layer_details[len(layer_type)+1:-1]  # Remove the layer type and parentheses

#             # Append the information to the list
#             layer_info_list.append({
#                 "Type": layer_type,
#                 "Parameters": layer_parameters
#             })

def extract_layer_info(model):
    """
    Extracts layer information from the given PyTorch model and returns it as a pandas DataFrame.
    
    Args:
        model: A PyTorch model (e.g., torchvision.models.ResNet50)
    
    Returns:
        pd.DataFrame: A DataFrame containing the name, type, and parameters of each layer.
    """

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

            # Append the information to the list, including the layer name
            layer_info_list.append({
                "Name": name,
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

    return df



# def parse_model_and_weights():
#     """
#     Parse command-line arguments.
    
#     Returns:
#         argparse.Namespace: Parsed arguments.
#     """
#     parser = argparse.ArgumentParser(description='Load a model and its weights.')
#     parser.add_argument('--model', type=str, required=True,
#                         help='Name of the model to load (e.g., "resnet50", "vgg16").')
#     parser.add_argument('--weights', type=str, required=True,
#                         help='Name of the weights class to load (e.g., "ResNet50_Weights", "VGG16_Weights").')

#     return parser.parse_args()
def parse_model_and_weights():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Load a model, weights, and input size.')
    
    # Model argument
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the model to load (e.g., "resnet50", "vgg16").')
    
    # Weights argument
    parser.add_argument('--weights', type=str, required=True,
                        help='Name of the weights class to load (e.g., "ResNet50_Weights", "VGG16_Weights").')
    
    # Input size argument (expects a tuple of integers, e.g., 3,224,224)
    parser.add_argument('--input_size', type=lambda s: tuple(map(int, s.split(','))), required=True,
                        help='Input size as a tuple of three integers (e.g., 3,224,224).')

    return parser.parse_args()







def process_model(input_size=(3, 224, 224), filter_types=['Conv2d', 'Linear'], exclude_string='downsample'):
    args = parse_model_and_weights()
    model = get_model_and_weights(args.model, args.weights)
    print(f'Loaded model: {args.model} with weights class: {args.weights}')

    df = extract_layer_info(model)

    # Filter the DataFrame based on configurable types and exclude string
    filtered_df = df[df['Type'].isin(filter_types)]
    filtered_df = filtered_df[~filtered_df['Name'].str.contains(exclude_string)]
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df = filtered_df.drop(columns=['Name'])

    input_data = torch.randn(1, *input_size)

    torchinfo_writer = TorchinfoWriter(model, input_data=input_data, verbose=0)
    torchinfo_writer.construct_model_tree()

    df_bigtree = torchinfo_writer.get_dataframe()

    # Apply the same filters to the torchinfo writer DataFrame
    filtered_df_bigtree = df_bigtree[df_bigtree['Type'].isin(filter_types)]
    filtered_df_bigtree = filtered_df_bigtree[~filtered_df_bigtree.index.str.contains(exclude_string)]
    filtered_df_bigtree.reset_index(drop=True, inplace=True)
    filtered_df_bigtree = filtered_df_bigtree.drop(columns=['Name', 'Type'])

    # Combine filtered DataFrames
    df = filtered_df.join(filtered_df_bigtree)

    df = df.rename(columns={0: 'input_channels', 1: 'output_channels', 2: 'kernel_size', 3: 'stride', 4: 'padding', 5: 'bias'})
    df = df.drop(columns=['MACs', 'Parameters'])

    word_to_find = 'bias'
    replacement_entry = 'padding=(0, 0)'

    # Move entries containing the word to 'other' column and replace with 'replacement entry'
    # df['bias'] = df['padding'].apply(lambda x: x if word_to_find in x else None)
    # df['padding'] = df['column1'].apply(lambda x: replacement_entry if word_to_find in x else x)
    # Use str.contains to identify rows that contain the specific word
    df['bias'] = df['padding'].where(df['padding'].str.contains(word_to_find, na=False))

    # Replace the matching entries in 'column1' with the replacement entry
    # df['padding'] = df['padding'].where(~df['padding'].str.contains(word_to_find, na=True), replacement_entry)
    df.loc[
    (df['Type'] == 'Conv2d') & (df['padding'].str.contains(word_to_find, na=False)),
    'padding'
] = replacement_entry


    # Adjust column data types
    df['Input Size'] = df['Input Size'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df['Kernel Size'] = df['Kernel Size'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df['Output Size'] = df['Output Size'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Group by all columns and count occurrences
    df_counts = df.groupby(list(df.columns), dropna=False).size().reset_index(name='count')

    # print(df_counts)

    total_count = df_counts['count'].sum()

    assert len(df) == total_count, "Error: The number of layers does not match the sum of all counted layers!"


    return(df_counts)


