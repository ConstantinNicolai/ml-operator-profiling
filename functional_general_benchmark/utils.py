import argparse
import importlib
import torch
from torchvision import models, transforms
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import re
import pandas as pd
import numpy as np
import os
from datetime import datetime
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter

from collections import defaultdict
import pickle
import lzma
import yaml
import time
import math

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


def process_log_file(in_file, iterations):
    try:
        # Load the log file into a pandas DataFrame
        df = pd.read_csv(in_file, delimiter=',', on_bad_lines='skip', header=None)

        # print(df)

        # Assign column names
        df.columns = ['Timestamp', 'Value1', 'Value2', 'Value3', 'Value4']

        # Drop any rows that contain NaN values
        df = df.dropna()

        # Remove any leading/trailing whitespace
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Convert the 'Timestamp' column to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # Drop rows where 'Timestamp' could not be converted (still NaT after coercion)
        df = df.dropna(subset=['Timestamp'])

        # Convert the other columns to numeric, coercing errors into NaNs
        df[['Value1', 'Value2', 'Value3', 'Value4']] = df[['Value1', 'Value2', 'Value3', 'Value4']].apply(pd.to_numeric, errors='coerce')

        # Drop any rows that now contain NaN values
        df = df.dropna()

        # Calculate the time difference between the first and last timestamp in seconds
        time_difference_seconds = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds()

        # Exclude the last row for calculations
        df_without_last = df.iloc[:-1]

        # Calculate the standard deviation for Value2 before filtering
        std_value2 = df_without_last['Value2'].std()

        # Filter out values outside the 3 standard deviation range for Value2
        mean_value2 = df_without_last['Value2'].mean()
        filtered_df = df_without_last[
            (np.abs(df_without_last['Value2'] - mean_value2) <= 3 * std_value2)
        ].copy()

        # Calculate the filtered mean for Value2
        filtered_mean_value2 = filtered_df['Value2'].mean()

        filtered_mean_value2_error = 5 / math.sqrt(len(df))

        # Calculate the total energy in joules (energy = power * time)
        total_energy_joules = filtered_mean_value2 * time_difference_seconds

        total_energy_joules_error = filtered_mean_value2_error * time_difference_seconds

        # Calculate the energy per iteration
        energy_per_iteration_in_milli_joule = 1000 * (total_energy_joules / iterations)

        energy_per_iteration_in_milli_joule_error = 1000 * (total_energy_joules_error / iterations)

        time_per_iteration = time_difference_seconds / iterations

        return (iterations, 
                time_difference_seconds, 
                time_per_iteration, 
                filtered_mean_value2, 
                std_value2, 
                total_energy_joules, 
                energy_per_iteration_in_milli_joule,
                total_energy_joules_error,
                energy_per_iteration_in_milli_joule_error)

    except Exception as e:
        print(f"Error processing the log file: {e}")
        return None



def process_log_file_with_time_jitter_uncertainty(in_file, iterations):
    try:
        # Load the log file
        df = pd.read_csv(in_file, delimiter=',', on_bad_lines='skip', header=None)

        # Assign column names
        df.columns = ['Timestamp', 'Value1', 'Value2', 'Value3', 'Value4']
        df = df.dropna()

        # Convert the 'Timestamp' column to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        df = df.dropna(subset=['Timestamp'])

        # Calculate the total time difference between the first and last timestamps
        time_difference_seconds = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds()

        # Add uncertainties to Value2 (±5 as per manufacturer)
        df['Value2_with_uncertainty'] = df['Value2'].apply(lambda x: ufloat(x, 5))

        # Calculate the mean of Value2 with uncertainty
        mean_value2_with_uncertainty = df['Value2_with_uncertainty'].mean()

        # Calculate total energy in joules (energy = power * time) with uncertainty
        total_energy_joules_with_uncertainty = mean_value2_with_uncertainty * time_difference_seconds_with_uncertainty

        # Calculate energy per iteration with uncertainty
        energy_per_iteration_with_uncertainty = total_energy_joules_with_uncertainty / iterations

        # Convert energy per iteration to millijoules with uncertainty
        energy_per_iteration_in_milli_joule_with_uncertainty = 1000 * energy_per_iteration_with_uncertainty

        # Return calculated values with uncertainty
        return iterations, time_difference_seconds_with_uncertainty, mean_value2_with_uncertainty, total_energy_joules_with_uncertainty, energy_per_iteration_in_milli_joule_with_uncertainty

    except Exception as e:
        print(f"Error processing the log file: {e}")
        return None


# DATASET_DIR = "dataset_history/"

# Get the latest dataset file
def get_latest_dataset_file(DATASET_DIR):
    files = [f for f in os.listdir(DATASET_DIR) if f.startswith("dataset_") and f.endswith(".pt")]
    if not files:
        return None
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(DATASET_DIR, f)))
    return os.path.join(DATASET_DIR, latest_file)

# Load the latest dataset
def load_latest_dataset(DATASET_DIR):
    latest_file = get_latest_dataset_file(DATASET_DIR)
    if latest_file:
        print(f"Loading dataset from {latest_file}")
        return torch.load(latest_file)
    else:
        print("No dataset found, initializing new dataset")
        return []

# Save the dataset with a timestamp
def save_dataset(dataset, DATASET_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_{timestamp}.pt"
    filepath = os.path.join(DATASET_DIR, filename)
    torch.save(dataset, filepath)
    print(f"Dataset saved to {filepath}")