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




###############
from datetime import datetime

runtimes = []
wattages = []

def load_latest_dataset_from_dir(clockspeed_dir, clockspeed_label):
    # Get all .pt files in the directory
    files = [f for f in os.listdir(clockspeed_dir) if f.startswith("dataset_") and f.endswith(".pt")]

    # Extract datetime from filenames and sort
    def get_timestamp(file):
        try:
            date_str = file.split("_")[1] + "_" + file.split("_")[2].split(".")[0]
            return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        except Exception:
            return None

    # Filter out files that couldn't be parsed
    files_with_timestamps = [(f, get_timestamp(f)) for f in files]
    files_with_timestamps = [ft for ft in files_with_timestamps if ft[1] is not None]

    if not files_with_timestamps:
        raise ValueError(f"No valid dataset files found in {clockspeed_dir}")

    # Sort by datetime (latest first)
    latest_file = sorted(files_with_timestamps, key=lambda x: x[1], reverse=True)[0][0]
    full_path = os.path.join(clockspeed_dir, latest_file)
    print(f"Loading most recent dataset for {clockspeed_label}: {full_path}")

    # Load the dataset
    dataset = torch.load(full_path, map_location=torch.device('cpu'))
    dataset_list = [list(item) for item in dataset]

    layers = [row[0] for row in dataset_list]
    input_sizes = [row[1] for row in dataset_list]
    # runtimes = [row[16] for row in dataset_list]
    # wattages = [row[21] for row in dataset_list]
    runtimes.extend([row[2] for row in dataset_list])
    wattages.extend([row[8] for row in dataset_list])
    

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

    # Load the encoder from the file
    with open('onehot_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    # 2. Fit and transform the column (reshape is needed because we expect a 2D array)
    onehot_encoded = encoder.transform(df[["type"]])

    # 3. Convert to DataFrame with proper column names (using encoder.categories_)
    onehot_df = pd.DataFrame(onehot_encoded.astype(bool),
                            columns=encoder.get_feature_names_out(["type"]),
                            index=df.index)

    # Drop the original column and add the new one-hot encoded columns
    df = df.drop("type", axis=1)

    df = pd.concat([df, onehot_df], axis=1)

    # Replace NaN values with -1
    df = df.replace(np.nan, -1)

    df = replace_applicable_flags(df)

    df = add_input_sizes_with_flags_to_df(df, input_sizes)

    df['clockspeed'] = clockspeed_label

    return df

# Define your clockspeed directories
clockspeed_dirs = {
    1440: "../functional_general_benchmark/datasets_train/dataset_history_A30_1440",
    1200: "../functional_general_benchmark/datasets_train/dataset_history_A30_1200",
    900: "../functional_general_benchmark/datasets_train/dataset_history_A30_900",
    600: "../functional_general_benchmark/datasets_train/dataset_history_A30_600",
    300: "../functional_general_benchmark/datasets_train/dataset_history_A30_300",
    210: "../functional_general_benchmark/datasets_train/dataset_history_A30_210",
}

# Load and merge all datasets
dataframes = []
for label, dir_path in clockspeed_dirs.items():
    df = load_latest_dataset_from_dir(dir_path, label)
    dataframes.append(df)

# Combine into one dataframe
final_df = pd.concat(dataframes, ignore_index=True)

input_features = final_df.to_numpy()

print(input_features.shape)

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




##################################################

import matplotlib.pyplot as plt
import random
import numpy as np


turquoise_color = '#2598be'
maroon_color = '#BE254D'

# Select a random subset of 10 test samples
num_samples = 10
random_indices = [26,27,28,29,30,31,32,33,34,35]

# Get the corresponding ground truth and predictions
true_runtimes = 1000*y_test_runtime[random_indices]
pred_runtimes = 1000*y_pred_runtime[random_indices]

true_power = y_test_wattage[random_indices]
pred_power = y_pred_wattage[random_indices]

# Compute energy consumption (runtime * power)
true_energy = true_runtimes * true_power
pred_energy = pred_runtimes * pred_power

# Load the encoder from the file
with open('onehot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Identify the one-hot encoded columns within X_test
num_original_features = X_test.shape[1] - len(encoder.get_feature_names_out(["type"]))
onehot_encoded_test = X_test[:, -21:-9]

# Inverse transform to get back original layer types
layer_types = encoder.inverse_transform(onehot_encoded_test)

# Extract input sizes for selected samples
input_sizes = X_test[random_indices][:, [-2, -4, -6, -8]]

# Generate formatted labels, removing -1 values
labels = []
for layer, sizes in zip(layer_types[random_indices].flatten(), input_sizes):
    valid_sizes = [str(int(size)) for size in sizes if size != -1]
    label = f"{layer} ({'x'.join(valid_sizes)})"
    labels.append(label)

# Set x-axis labels for plots
x_labels = labels
x = range(num_samples)

# Runtime comparison plot
plt.figure(figsize=(15, 10))
plt.bar(x, true_runtimes, width=0.4, label="True Runtime", alpha=0.7, color=turquoise_color)
plt.bar([i + 0.4 for i in x], pred_runtimes, width=0.4, label="Predicted Runtime", alpha=0.7, color=maroon_color)
plt.xticks([i + 0.2 for i in x], x_labels, rotation=45, ha="right")
plt.title("Runtime Prediction vs Ground Truth")
plt.ylabel("Runtime [ms]")
plt.legend()
plt.tight_layout()
plt.savefig('runtime_plot.png', format='png')
plt.savefig('runtime_plot.pdf', format='pdf')
plt.close()

# Power comparison plot
plt.figure(figsize=(15, 10))
plt.bar(x, true_power, width=0.4, label="True Power", alpha=0.7, color="orange")
plt.bar([i + 0.4 for i in x], pred_power, width=0.4, label="Predicted Power", alpha=0.7, color="purple")
plt.xticks([i + 0.2 for i in x], x_labels, rotation=45, ha="right")
plt.title("Power Prediction vs Ground Truth")
plt.ylabel("Power [W]")
plt.legend()
plt.tight_layout()
plt.savefig('power_plot.png', format='png')
plt.savefig('power_plot.pdf', format='pdf')
plt.close()

# Energy consumption comparison plot
plt.figure(figsize=(15, 10))
plt.bar(x, true_energy, width=0.4, label="True Energy Consumption", alpha=0.7, color="green")
plt.bar([i + 0.4 for i in x], pred_energy, width=0.4, label="Predicted Energy Consumption", alpha=0.7, color="red")
plt.xticks([i + 0.2 for i in x], x_labels, rotation=45, ha="right")
plt.title("Energy Consumption Prediction vs Ground Truth")
plt.ylabel("Energy Consumption [mJ]")
plt.legend()
plt.tight_layout()
plt.savefig('energy_plot.png', format='png')
plt.savefig('energy_plot.pdf', format='pdf')
plt.close()






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
joblib.dump(runtime_model, 'model_dump/runtime_model_clockinput_inf.pkl')
joblib.dump(wattage_model, 'model_dump/wattage_model_clockinput_inf.pkl')

