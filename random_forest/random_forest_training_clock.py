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
    runtimes.extend([row[16] for row in dataset_list])
    wattages.extend([row[21] for row in dataset_list])
    

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



# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from xgboost import XGBRegressor  # Import XGBoost
# from sklearn.neural_network import MLPRegressor  # Import MLP


# def get_model(model_type, n_estimators=100, criterion='squared_error', random_state=42):
#     if model_type == 'random_forest':
#         return RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, random_state=random_state)
#     elif model_type == 'extra_trees':
#         return ExtraTreesRegressor(n_estimators=n_estimators, criterion=criterion, random_state=random_state)
#     elif model_type == 'xgboost':
#         return XGBRegressor(n_estimators=n_estimators, objective='reg:squarederror', random_state=random_state)
#     elif model_type == 'mlp':
#         return MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=random_state)
#     else:
#         raise ValueError("Unsupported model type. Choose 'random_forest', 'extra_trees', 'xgboost', or 'mlp'.")


# # Assuming you have 'input_features' for your inputs and 'runtimes' and 'wattages' for your targets
# # input_features = df.to_numpy()
# # runtimes = [row[2] for row in dataset_list]
# # wattages = [row[8] for row in dataset_list]

# runtime_min, runtime_max = np.min(runtimes), np.max(runtimes)
# wattage_min, wattage_max = np.min(wattages), np.max(wattages)

# train_indices = np.where((runtimes == runtime_min) | (runtimes == runtime_max) |
#                          (wattages == wattage_min) | (wattages == wattage_max))[0]

# remaining_indices = np.setdiff1d(np.arange(len(runtimes)), train_indices)
# train_remaining, test_remaining = train_test_split(remaining_indices, test_size=0.2, random_state=42)
# train_indices = np.concatenate([train_indices, train_remaining])

# X_train, X_test = input_features[train_indices], input_features[test_remaining]
# y_train_runtime, y_test_runtime = np.array(runtimes)[train_indices], np.array(runtimes)[test_remaining]
# y_train_wattage, y_test_wattage = np.array(wattages)[train_indices], np.array(wattages)[test_remaining]

# # Choose the model type here
# model_type_runtime = 'random_forest'   # 'random_forest' or 'extra_trees'
# model_type_wattage = 'random_forest' # 'random_forest' or 'extra_trees'

# # Create and train models
# runtime_model = get_model(model_type_runtime, n_estimators=900, criterion='squared_error')
# runtime_model.fit(X_train, y_train_runtime)

# wattage_model = get_model(model_type_wattage, n_estimators=100, criterion='squared_error')
# wattage_model.fit(X_train, y_train_wattage)

# # Predictions
# y_pred_runtime = runtime_model.predict(X_test)
# y_pred_wattage = wattage_model.predict(X_test)
# energy_pred = y_pred_runtime * y_pred_wattage

# # Metrics
# runtime_mse = mean_squared_error(y_test_runtime, y_pred_runtime)
# wattage_mse = mean_squared_error(y_test_wattage, y_pred_wattage)
# r2_runtime = r2_score(y_test_runtime, y_pred_runtime)
# r2_wattage = r2_score(y_test_wattage, y_pred_wattage)

# # Output
# print(f"Test MSE for Runtime Prediction: {runtime_mse:.4f}")
# print(f"Test MSE for Wattage Prediction: {wattage_mse:.4f}")
# print(f"R² for Runtime Prediction: {r2_runtime:.4f}")
# print(f"R² for Wattage Prediction: {r2_wattage:.4f}")
# print(f"Sample Predicted Runtimes: {y_pred_runtime[:5]}")
# print(f"Sample Predicted Wattages: {y_pred_wattage[:5]}")
# print(f"Sample Energy Predictions: {energy_pred[:5]}")



#####################################################


import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score  # Added cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

def get_model(model_type, n_estimators=100, criterion='squared_error', random_state=6447):
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

# --- Your existing data splitting logic ---
runtime_min, runtime_max = np.min(runtimes), np.max(runtimes)
wattage_min, wattage_max = np.min(wattages), np.max(wattages)

train_indices = np.where((runtimes == runtime_min) | (runtimes == runtime_max) |
                         (wattages == wattage_min) | (wattages == wattage_max))[0]

remaining_indices = np.setdiff1d(np.arange(len(runtimes)), train_indices)
train_remaining, test_remaining = train_test_split(remaining_indices, test_size=0.2, random_state=56353)
train_indices = np.concatenate([train_indices, train_remaining])

X_train, X_test = input_features[train_indices], input_features[test_remaining]
y_train_runtime, y_test_runtime = np.array(runtimes)[train_indices], np.array(runtimes)[test_remaining]
y_train_wattage, y_test_wattage = np.array(wattages)[train_indices], np.array(wattages)[test_remaining]

# --- Model Selection ---
model_type_runtime = 'random_forest'
model_type_wattage = 'random_forest'

# --- Cross-Validation ---  
def evaluate_with_cv(model, X, y, model_name):
    cv_scores = cross_val_score(model, X, y, cv=15, scoring='r2')
    print(f"{model_name} CV R² scores: {cv_scores}")
    print(f"{model_name} Mean CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n")

# Train and evaluate runtime model
runtime_model = get_model(model_type_runtime, n_estimators=100, criterion='squared_error')
# old num estimaros is 900
evaluate_with_cv(runtime_model, X_train, y_train_runtime, "Runtime Model")
runtime_model.fit(X_train, y_train_runtime)
y_pred_runtime = runtime_model.predict(X_test)

# Train and evaluate wattage model
wattage_model = get_model(model_type_wattage, n_estimators=100, criterion='squared_error')
evaluate_with_cv(wattage_model, X_train, y_train_wattage, "Wattage Model")
wattage_model.fit(X_train, y_train_wattage)
y_pred_wattage = wattage_model.predict(X_test)

# --- NEW: Feature Importance Correlation (Added) ---
if model_type_runtime == 'random_forest' or model_type_runtime == 'xgboost':
    # Get feature importances from both model types
    xgb_temp = get_model('xgboost', n_estimators=100).fit(X_train, y_train_runtime)
    xgb_temp_watt = get_model('xgboost', n_estimators=100).fit(X_train, y_train_wattage)

    evaluate_with_cv(xgb_temp, X_train, y_train_runtime, "Runtime Model XGB")

    evaluate_with_cv(xgb_temp_watt, X_train, y_train_wattage, "Wattage Model XGB")

    rf_importances = runtime_model.feature_importances_
    xgb_importances = xgb_temp.feature_importances_
    corr, _ = spearmanr(rf_importances, xgb_importances)

    rf_importances_watt = wattage_model.feature_importances_
    xgb_importances_watt = xgb_temp_watt.feature_importances_
    corr1, _ = spearmanr(rf_importances_watt, xgb_importances_watt)

    print(f"\nFeature Importance Correlation (Random Forest vs XGBoost) runtime: {corr:.3f}")
    print(f"\nFeature Importance Correlation (Random Forest vs XGBoost) wattage: {corr1:.3f}")


    y_pred_runtime_xgb = xgb_temp.predict(X_test)
    y_pred_wattage_xgb = xgb_temp_watt.predict(X_test)

    r2_runtime_xgb = r2_score(y_test_runtime, y_pred_runtime_xgb)
    r2_wattage_xgb = r2_score(y_test_wattage, y_pred_wattage_xgb)

    print(f"Test R² for Runtime Prediction XGB: {r2_runtime_xgb:.4f}")
    print(f"Test R² for Wattage Prediction XGB: {r2_wattage_xgb:.4f}")


# --- Rest of your original code (unchanged) ---
energy_pred = y_pred_runtime * y_pred_wattage

# Metrics
runtime_mse = mean_squared_error(y_test_runtime, y_pred_runtime)
wattage_mse = mean_squared_error(y_test_wattage, y_pred_wattage)
r2_runtime = r2_score(y_test_runtime, y_pred_runtime)
r2_wattage = r2_score(y_test_wattage, y_pred_wattage)




# Output
print("\n--- Test Set Evaluation ---")
print(f"Test MSE for Runtime Prediction: {runtime_mse:.4f}")
print(f"Test MSE for Wattage Prediction: {wattage_mse:.4f}")
print(f"Test R² for Runtime Prediction: {r2_runtime:.4f}")
print(f"Test R² for Wattage Prediction: {r2_wattage:.4f}")
print(f"Sample Predicted Runtimes: {y_pred_runtime[:5]}")
print(f"Sample Predicted Wattages: {y_pred_wattage[:5]}")
print(f"Sample Energy Predictions: {energy_pred[:5]}")




import matplotlib.pyplot as plt
import random
import numpy as np




plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 17
})





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
input_sizes = X_test[random_indices][:, [-3, -5, -7, -9]]

print("here!")
print(input_sizes)

clock_speeds_test_set = X_test[random_indices][:, [-1]]

# Generate formatted labels, removing -1 values
labels = []
for layer, sizes, speeds in zip(layer_types[random_indices].flatten(), input_sizes, clock_speeds_test_set):
    #print(speeds[0])
    #speed_entries = [str(int(speed)) for speed in speeds if speed != -1]
    valid_sizes = [str(int(size)) for size in sizes if size != -1]
    label = f"{layer} ({'x'.join(valid_sizes)}) {speeds[0]} MHz"
    labels.append(label)

# Set x-axis labels for plots
x_labels = labels
x = range(num_samples)


# Create figure with 3 vertically stacked subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 24), sharex=True)

# Plot 1: Runtime comparison
ax1.bar(x, true_runtimes, width=0.4, label="Measured Runtime", alpha=0.7, color='#1f77b4')
ax1.bar([i + 0.4 for i in x], pred_runtimes, width=0.4, label="Predicted Runtime", alpha=0.7, color='#ff7f0e')
#ax1.set_title("Runtime Prediction vs Ground Truth")
ax1.set_ylabel("Runtime [ms]")
ax1.legend()


# Plot 2: Power comparison
ax2.bar(x, true_power, width=0.4, label="Measured Power", alpha=0.7, color='orange')
ax2.bar([i + 0.4 for i in x], pred_power, width=0.4, label="Predicted Power", alpha=0.7, color='purple')
#ax2.set_title("Power Prediction vs Ground Truth")
ax2.set_ylabel("Power [W]")
ax2.legend()


# Plot 3: Energy comparison (will show x-axis labels)
ax3.bar(x, true_energy, width=0.4, label="Measured Energy Consumption", alpha=0.7, color='green')
ax3.bar([i + 0.4 for i in x], pred_energy, width=0.4, label="Predicted Energy Consumption", alpha=0.7, color='red')
#ax3.set_title("Energy Consumption Prediction vs Ground Truth")
ax3.set_ylabel("Energy [mJ]")
ax3.legend()


# Only show x-axis labels on the bottom plot
ax3.set_xticks([i + 0.2 for i in x])
ax3.set_xticklabels(x_labels, rotation=45, ha="right")

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('stacked_comparison.png', format='png', dpi=300)
plt.savefig('stacked_comparison.pdf', format='pdf')





# print("###############################")

# import random

# # Select a random index from the training set
# random_index = random.choice(train_indices)

# # Extract the feature vector for the selected index
# feature_vector = input_features[random_index]

# # Display the selected feature vector
# print("Original Feature Vector from Training Set:")
# print(feature_vector)

# # Manually edit the feature vector
# # Example: Modify specific features manually
# feature_vector[27] = 83  # Edit feature at index 0
# # feature_vector[2] = 0.5  # Edit feature at index 2
# # You can manually modify any features as needed

# # Reshape the modified feature vector for prediction
# test_input = feature_vector.reshape(1, -1)

# # Use the trained models to make predictions on this edited input
# predicted_runtime = runtime_model.predict(test_input)
# predicted_wattage = wattage_model.predict(test_input)
# predicted_energy = predicted_runtime * predicted_wattage

# # Output predictions for the manually edited input
# print("\nPredictions for Manually Edited Input:")
# print(f"Modified Features: {feature_vector}")
# print(f"Predicted Runtime: {predicted_runtime[0]:.4f}")
# print(f"Predicted Wattage: {predicted_wattage[0]:.4f}")
# print(f"Predicted Energy Consumption: {predicted_energy[0]:.4f}")



import joblib

# Save models to files
joblib.dump(runtime_model, 'model_dump/runtime_model_clockinput_train.pkl')
joblib.dump(wattage_model, 'model_dump/wattage_model_clockinput_train.pkl')

