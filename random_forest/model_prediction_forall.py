import joblib
import lzma
import pickle
import pandas as pd
import numpy as np
import math
import glob
import os
import yaml
import argparse


# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="set the path for the dataset")
parser.add_argument("--clock", type=int, required=True, help="Specify the clock in MHz that you want predictions for")


# Parse arguments
args = parser.parse_args()
clock = args.clock


list_attemps = []
len_list = []


meas_dir_path = f"../predictions/A30"
meas_dir_path_to_be_added = f"../predictions/A30/"

for entry in os.listdir(meas_dir_path):
    filtered_files = glob.glob(meas_dir_path_to_be_added + entry + '/*.pkl.xz_filtered')

    if not filtered_files:
        print(f"No matching file found for {entry}")
        continue  # Skip to the next entry

    file_path = filtered_files[0]  # Take the first matching file

    # Open the compressed file correctly
    with lzma.open(file_path, 'rb') as file_:
        saved_dict = pickle.load(file_)


# with lzma.open('../predictions/A30/wide_resnet50_2_32,3,224,224/wide_resnet50_2_32_3_224_224.pkl.xz_filtered') as file_:
#     saved_dict = pickle.load(file_)

    list_attemps_section = list(saved_dict.items())

    len_list.append(len(list_attemps_section))

    list_attemps.extend(list_attemps_section)


    # for item in list_attemps:
    #     print(item)
    #     print(len(item))
    #     print(item[0])
    #     print(type(item[0]))
    #     print(item[1])
    #     print(type(item[1]))

    # for row in list_attemps:
    #     print(row[1][0])
    #     print(type(row[1][0]))


layers = [row[1][0] for row in list_attemps]
input_sizes = [row[0][-1] for row in list_attemps]



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

df['clockspeed'] = clock

# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)

# print(df)

# # Display the DataFrame
# print(df.to_numpy())

# print(df.to_numpy()[0:3])

input_features = df.to_numpy()

# for i in range(len(input_features)):
#     input_features[i]

# print(input_features[2])
# print(len(input_features[2]))

# input_features[2].reshape(1,-1)

# # Extract the feature vector for the selected index
# feature_vector = input_features[2]

# # Reshape the modified feature vector for prediction
# test_input = feature_vector.reshape(1, -1)
# Load models
runtime_model = joblib.load('model_dump/runtime_model.pkl')
wattage_model = joblib.load('model_dump/wattage_model.pkl')


for i in range(len(input_features)):


    # Example: Making predictions
    y_pred_runtime = runtime_model.predict(input_features[i].reshape(1,-1))
    y_pred_wattage = wattage_model.predict(input_features[i].reshape(1,-1))
    energy_pred = y_pred_runtime * y_pred_wattage


    # print(type(y_pred_runtime))
    list_attemps[i][1].append(list_attemps[i][0][-1])
    list_attemps[i][1].append(y_pred_runtime[0])
    list_attemps[i][1].append(y_pred_wattage[0])
    list_attemps[i][1].append(energy_pred[0])




result = [row[1] for row in list_attemps]


counter = 0

position_in_result = 0


for entry in os.listdir(meas_dir_path):
    with open(meas_dir_path_to_be_added + entry + '/summary.yml', 'r') as summ:
        config = yaml.safe_load(summ)

    print(config['model_name'], config['input_size'])

    old_position = position_in_result

    position_in_result = position_in_result + len_list[counter]

    result_section = result[old_position:position_in_result]

    # Initialize sums
    time_sum = 0
    energy_sum = 0
    # energy_error_squared_sum = 0
    # time_error_squared_sum = 0


    for item in result_section:
        count_of_this_layer = item[1]
        runtime = item[3]
        energy = item[5]
        # iterations = item[7]
        # runtime_for_all_iterations = item[8]
        # energy_error = 0
        # runtime_error = 0
        if math.isnan(runtime) == False:
            time_sum = time_sum + count_of_this_layer * runtime
        else:
            print("encountered nan value in runtime, incomplete sum")
        # if math.isnan(runtime_error) == False:
        #     runtime_error_squared_sum = runtime_error_squared_sum + count_of_this_layer * runtime_error * runtime_error
        # else:
        #     print("encountered nan value in runtime error, incomplete sum")
        if math.isnan(energy) == False:
            energy_sum = energy_sum + count_of_this_layer * energy
        else:
            print("encountered nan value in energy, incomplete sum")
        # if math.isnan(energy_error) == False:
        #     energy_error_squared_sum = energy_error_squared_sum + count_of_this_layer * energy_error * energy_error
        # else:
        #     print("encountered nan value in energy error, incomplete sum")

    print(1000*time_sum)
    # print(1000*math.sqrt(runtime_error_squared_sum), '[ms]')
    print(1000*energy_sum)
    # print(math.sqrt(energy_error_squared_sum), '[mJ]')




    
    counter = counter +1


# for row in result:
#     print(row)


# # Initialize sums
# time_sum = 0
# energy_sum = 0
# # energy_error_squared_sum = 0
# # time_error_squared_sum = 0


# for item in result:
#     count_of_this_layer = item[1]
#     runtime = item[3]
#     energy = item[5]
#     # iterations = item[7]
#     # runtime_for_all_iterations = item[8]
#     # energy_error = 0
#     # runtime_error = 0
#     if math.isnan(runtime) == False:
#         time_sum = time_sum + count_of_this_layer * runtime
#     else:
#         print("encountered nan value in runtime, incomplete sum")
#     # if math.isnan(runtime_error) == False:
#     #     runtime_error_squared_sum = runtime_error_squared_sum + count_of_this_layer * runtime_error * runtime_error
#     # else:
#     #     print("encountered nan value in runtime error, incomplete sum")
#     if math.isnan(energy) == False:
#         energy_sum = energy_sum + count_of_this_layer * energy
#     else:
#         print("encountered nan value in energy, incomplete sum")
#     # if math.isnan(energy_error) == False:
#     #     energy_error_squared_sum = energy_error_squared_sum + count_of_this_layer * energy_error * energy_error
#     # else:
#     #     print("encountered nan value in energy error, incomplete sum")

# print(1000*time_sum, "[ms]")
# # print(1000*math.sqrt(runtime_error_squared_sum), '[ms]')
# print(1000*energy_sum, "[mJ]")
# # print(math.sqrt(energy_error_squared_sum), '[mJ]')