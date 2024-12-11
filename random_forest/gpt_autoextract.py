import torch.nn as nn
import pandas as pd
import numpy as np

# Example PyTorch objects (test layers as specified)
layers = [
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
    nn.Linear(100, 10, bias=False),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64, affine=True, eps=1e-5, momentum=0.1),
    nn.Dropout(p=0.2),
    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
]

# Predefined list of attributes to consider
attributes_to_extract = [
    "bias", "in_channels", "out_channels", "kernel_size", "padding", "stride",
    "in_features", "out_features", "inplace", "affine", "eps", "momentum",
    "num_features", "p", "output_size"
]

# Helper function to preprocess tuples
def preprocess_tuple(attribute_value, length=2):
    """Flattens a tuple or provides default values for non-applicable attributes."""
    if isinstance(attribute_value, tuple):
        return list(attribute_value)[:length]  # Ensure fixed length
    return [-1] * length  # Default for non-applicable attributes


def replace_applicable_flags(df):
    # Iterate over columns and replace -1 with 0 for columns containing 'applicable'
    applicable_cols = [col for col in df.columns if 'applicable' in col]
    df[applicable_cols] = df[applicable_cols].replace(-1, 0)
    return df


# # Feature extraction function with custom bias handling
# def extract_features_with_flags(layer, attributes):
#     features = {"type": type(layer).__name__}  # Include layer type
#     for attr in attributes:
#         if attr == "bias":
#             # Custom handling for the `bias` attribute
#             if hasattr(layer, "bias"):
#                 bias_value = getattr(layer, "bias")
#                 if bias_value is None:
#                     features["bias"] = 0
#                     features["bias_applicable"] = 1
#                 else:
#                     features["bias"] = 1  # Bias is a tensor
#                     features["bias_applicable"] = 1
#             else:
#                 features["bias"] = -1
#                 features["bias_applicable"] = 0
#         elif hasattr(layer, attr):
#             value = getattr(layer, attr)
#             if isinstance(value, (int, float, bool)):
#                 features[attr] = int(value) if isinstance(value, bool) else value
#                 features[f"{attr}_applicable"] = 1
#             elif isinstance(value, (tuple, list)):
#                 flattened = preprocess_tuple(value)
#                 for i, v in enumerate(flattened):
#                     features[f"{attr}_{i}"] = v  # Flattened components
#                 features[f"{attr}_applicable"] = 1
#             else:
#                 features[attr] = -1
#                 features[f"{attr}_applicable"] = 0
#         else:
#             features[attr] = -1  # Placeholder for non-existent attributes
#             features[f"{attr}_applicable"] = 0
#     return features


def extract_features_with_flags(layer, attributes):
    features = {"type": type(layer).__name__}  # Include layer type
    for attr in attributes:
        if attr == "bias":
            # Custom handling for the `bias` attribute
            if hasattr(layer, "bias"):
                bias_value = getattr(layer, "bias")
                if bias_value is None:
                    features["bias"] = 0
                    features["bias_applicable"] = 1
                else:
                    features["bias"] = 1  # Bias is a tensor
                    features["bias_applicable"] = 1
            else:
                features["bias"] = -1
                features["bias_applicable"] = 0
        elif hasattr(layer, attr):
            value = getattr(layer, attr)
            if isinstance(value, (int, float, bool)):
                # Single-value case
                features[attr] = int(value) if isinstance(value, bool) else value
                features[f"{attr}_applicable"] = 1
            elif isinstance(value, (tuple, list)):
                # Tuple case: create separate fields for each component
                flattened = preprocess_tuple(value)
                for i, v in enumerate(flattened):
                    features[f"{attr}_{i}"] = v  # Flattened components
                # Add a tuple-level flag
                features[f"{attr}_tuple_applicable"] = 1
            else:
                features[attr] = -1
                features[f"{attr}_applicable"] = 0
        else:
            features[attr] = -1  # Placeholder for non-existent attributes
            features[f"{attr}_applicable"] = 0
    return features




# Extract features for all layers
feature_list = [extract_features_with_flags(layer, attributes_to_extract) for layer in layers]

# Convert to DataFrame
df = pd.DataFrame(feature_list)

# One-hot encode the layer type
df = pd.get_dummies(df, columns=["type"], prefix="type")

# Replace NaN values with -1
df = df.replace(np.nan, -1)

df = replace_applicable_flags(df)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)

print(df)

# Display the DataFrame
print(df.to_numpy())
