import argparse
import importlib
import torch
from torchvision import models, transforms
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import re


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
        pd.DataFrame: A DataFrame containing the type and parameters of each layer.
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

    return df