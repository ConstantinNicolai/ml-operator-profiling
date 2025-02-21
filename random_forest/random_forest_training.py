import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os
import yaml
import lzma
import pickle
import math
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the saved .pt file
dataset = torch.load('../functional_general_benchmark/datasets_train/dataset_history_A30/dataset_20250213_132513.pt', map_location=torch.device('cpu'))


dataset_list = [list(item) for item in dataset]


print(dataset_list[4])


# # Example PyTorch objects (test layers as specified)
# layers = [
#     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
#     nn.Linear(100, 10, bias=False),
#     nn.ReLU(inplace=True),
#     nn.BatchNorm2d(64, affine=True, eps=1e-5, momentum=0.1),
#     nn.Dropout(p=0.2),
#     nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#     nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
# ]

layers = [row[0] for row in dataset_list]
input_sizes = [row[1] for row in dataset_list]
runtimes = [row[16] for row in dataset_list]
wattages = [row[21] for row in dataset_list]

# print(input_sizes[0:6])
# print(runtimes[0:6])
# print(wattages[0:6])

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

def add_input_sizes_with_flags_to_df(df, input_sizes):
    # Ensure the length of input_sizes matches the number of rows in the DataFrame
    assert len(input_sizes) == len(df), "input_sizes must have the same length as the DataFrame"
    
    # Iterate through each layer's input size
    for i, size in enumerate(input_sizes):
        # Extract the individual components, with fallback to -1 for missing values
        input_size_0 = size[0] if len(size) > 0 else -1
        input_size_1 = size[1] if len(size) > 1 else -1
        input_size_2 = size[2] if len(size) > 2 else -1
        input_size_3 = size[3] if len(size) > 3 else -1
        
        # Add input sizes as columns
        df.at[i, 'input_size_0'] = input_size_0
        df.at[i, 'input_size_1'] = input_size_1
        df.at[i, 'input_size_2'] = input_size_2
        df.at[i, 'input_size_3'] = input_size_3
        
        # Add applicability flags directly after their corresponding input size
        df.at[i, 'input_size_0_applicable'] = 1 if input_size_0 != -1 else 0
        df.at[i, 'input_size_1_applicable'] = 1 if input_size_1 != -1 else 0
        df.at[i, 'input_size_2_applicable'] = 1 if input_size_2 != -1 else 0
        df.at[i, 'input_size_3_applicable'] = 1 if input_size_3 != -1 else 0
    
    # Reorder columns to ensure applicability flags follow the corresponding input sizes
    column_order = []
    for col in df.columns:
        if 'input_size' in col and '_applicable' not in col:
            column_order.append(col)
            column_order.append(col + '_applicable')
        elif 'input_size' not in col:  # Keep other columns as is
            column_order.append(col)

    df = df[column_order]
    
    return df


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

df = add_input_sizes_with_flags_to_df(df, input_sizes)

# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)

# print(df)

# # Display the DataFrame
# print(df.to_numpy())

# print(df.to_numpy()[0:3])

input_features = df.to_numpy()


###############################################################

print(input_features[0])



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor  # Import XGBoost
from sklearn.neural_network import MLPRegressor  # Import MLP


def get_model(model_type, n_estimators=100, criterion='squared_error', random_state=42):
    if model_type == 'random_forest':
        return RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, random_state=random_state)
    elif model_type == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=n_estimators, criterion=criterion, random_state=random_state)
    elif model_type == 'xgboost':
        return XGBRegressor(n_estimators=n_estimators, objective='reg:squarederror', random_state=random_state)
    elif model_type == 'mlp':
        return MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=random_state)
    else:
        raise ValueError("Unsupported model type. Choose 'random_forest', 'extra_trees', 'xgboost', or 'mlp'.")


# Assuming you have 'input_features' for your inputs and 'runtimes' and 'wattages' for your targets
# input_features = df.to_numpy()
# runtimes = [row[2] for row in dataset_list]
# wattages = [row[8] for row in dataset_list]

runtime_min, runtime_max = np.min(runtimes), np.max(runtimes)
wattage_min, wattage_max = np.min(wattages), np.max(wattages)

train_indices = np.where((runtimes == runtime_min) | (runtimes == runtime_max) |
                         (wattages == wattage_min) | (wattages == wattage_max))[0]

remaining_indices = np.setdiff1d(np.arange(len(runtimes)), train_indices)
train_remaining, test_remaining = train_test_split(remaining_indices, test_size=0.2, random_state=42)
train_indices = np.concatenate([train_indices, train_remaining])

X_train, X_test = input_features[train_indices], input_features[test_remaining]
y_train_runtime, y_test_runtime = np.array(runtimes)[train_indices], np.array(runtimes)[test_remaining]
y_train_wattage, y_test_wattage = np.array(wattages)[train_indices], np.array(wattages)[test_remaining]

# Choose the model type here
model_type_runtime = 'random_forest'   # 'random_forest' or 'extra_trees'
model_type_wattage = 'random_forest' # 'random_forest' or 'extra_trees'

# Create and train models
runtime_model = get_model(model_type_runtime, n_estimators=900, criterion='absolute_error')
runtime_model.fit(X_train, y_train_runtime)

wattage_model = get_model(model_type_wattage, n_estimators=100, criterion='squared_error')
wattage_model.fit(X_train, y_train_wattage)

# Predictions
y_pred_runtime = runtime_model.predict(X_test)
y_pred_wattage = wattage_model.predict(X_test)
energy_pred = y_pred_runtime * y_pred_wattage

# Metrics
runtime_mse = mean_squared_error(y_test_runtime, y_pred_runtime)
wattage_mse = mean_squared_error(y_test_wattage, y_pred_wattage)
r2_runtime = r2_score(y_test_runtime, y_pred_runtime)
r2_wattage = r2_score(y_test_wattage, y_pred_wattage)

# Output
print(f"Test MSE for Runtime Prediction: {runtime_mse:.4f}")
print(f"Test MSE for Wattage Prediction: {wattage_mse:.4f}")
print(f"R² for Runtime Prediction: {r2_runtime:.4f}")
print(f"R² for Wattage Prediction: {r2_wattage:.4f}")
print(f"Sample Predicted Runtimes: {y_pred_runtime[:5]}")
print(f"Sample Predicted Wattages: {y_pred_wattage[:5]}")
print(f"Sample Energy Predictions: {energy_pred[:5]}")


print("###############################")

import random

# Select a random index from the training set
random_index = random.choice(train_indices)

# Extract the feature vector for the selected index
feature_vector = input_features[random_index]

# Display the selected feature vector
print("Original Feature Vector from Training Set:")
print(feature_vector)

# Manually edit the feature vector
# Example: Modify specific features manually
feature_vector[27] = 83  # Edit feature at index 0
# feature_vector[2] = 0.5  # Edit feature at index 2
# You can manually modify any features as needed

# Reshape the modified feature vector for prediction
test_input = feature_vector.reshape(1, -1)

# Use the trained models to make predictions on this edited input
predicted_runtime = runtime_model.predict(test_input)
predicted_wattage = wattage_model.predict(test_input)
predicted_energy = predicted_runtime * predicted_wattage

# Output predictions for the manually edited input
print("\nPredictions for Manually Edited Input:")
print(f"Modified Features: {feature_vector}")
print(f"Predicted Runtime: {predicted_runtime[0]:.4f}")
print(f"Predicted Wattage: {predicted_wattage[0]:.4f}")
print(f"Predicted Energy Consumption: {predicted_energy[0]:.4f}")



import joblib

# Save models to files
joblib.dump(runtime_model, 'model_dump/runtime_model.pkl')
joblib.dump(wattage_model, 'model_dump/wattage_model.pkl')

