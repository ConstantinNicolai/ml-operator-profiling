import torch
import os
import yaml
import lzma
import pickle
import math

# Load the saved .pt file

# This sets the dataset of operations used to sum up from

dataset = torch.load('datasets_train/dataset_history_A30/dataset_20250213_132513.pt', map_location=torch.device('cpu'))


# Just converting the entries into lists

dataset_list = [list(item) for item in dataset]

# Looping through all model input combinations tested , which are found in the measurements directory

for entry in os.listdir('./../measurements/A30'):
    with open('./../measurements/A30/' + entry + '/summary.yml', 'r') as file:
        config = yaml.safe_load(file)

    config['input_size'] = tuple(config['input_size'])
       
    tuple_str = "_".join(map(str, config['input_size']))
    filename = f"{config['model_name']}_{tuple_str}.pkl.xz"

# Loading the dataset of operations and number of occurences in the given model

    with lzma.open('./../measurements/A30/' + entry + '/' + filename + '_filtered') as file_:
        saved_dict = pickle.load(file_)

# Converting the dataset of operations and number of occurences N into a list 

    list_attemps = list(saved_dict.items())

    print(config['model_name'], config['input_size'])

# Sorting the entries we need from the model side dataset for comparison
    working_list = [item[1] + [item[0][2]] for item in list_attemps]


# Comparing both the full master dataset of operations and the one from the model inclusing the number
# of occurences N. Filtering through both and comparing them to merge into one dataset that can be used to 
# sum the operation level measurements for the current model. 
# A few activation functions had a problem with the extra representation check, so we skip it for these

    # Add missing entries from the master list to the list to modify, while preserving existing ones
    for item in working_list:
        for master_item in dataset_list:
            if master_item[0]._get_name() == "ReLU" or master_item[0]._get_name() == "SiLU":
                if item[0]._get_name() == master_item[0]._get_name() and item[2] == master_item[1] and len(item) == 3:    # and len(item) == 3   # module._get_name(), module.extra_repr()
                    # Check which items from master_item are missing in item and extend it
                    o = 0
                    for entry in master_item[1:]:
                        o = o +1
                        if o != 1:
                            item.append(entry)
            else:
                if item[0]._get_name() == master_item[0]._get_name() and item[0].extra_repr() == master_item[0].extra_repr() and item[2] == master_item[1] and len(item) == 3:    # and len(item) == 3   # module._get_name(), module.extra_repr()
                    # Check which items from master_item are missing in item and extend it
                    o = 0
                    for entry in master_item[1:]:
                        o = o +1
                        if o != 1:
                            item.append(entry)

    time_sum = 0
    energy_sum = 0
    energy_error_squared_sum = 0
    runtime_error_squared_sum = 0


# The entry where each measurement lies is counted from printing out and entry of working_list
# and can be checked by doing just that

    # print(working_list[0])

    for item in working_list:
        count_of_this_layer = item[1]
        runtime = item[17]
        energy = item[19]
        iterations = item[7]
        runtime_for_all_iterations = item[8]
        energy_error = item[20]
        runtime_error = item[18]
        if math.isnan(runtime) == False:
            time_sum = time_sum + count_of_this_layer * runtime
        else:
            print("encountered nan value in runtime, incomplete sum")
        if math.isnan(runtime_error) == False:
            runtime_error_squared_sum = runtime_error_squared_sum + count_of_this_layer * runtime_error * runtime_error
        else:
            print("encountered nan value in runtime error, incomplete sum")
        if math.isnan(energy) == False:
            energy_sum = energy_sum + count_of_this_layer * energy
        else:
            print("encountered nan value in energy, incomplete sum")
        if math.isnan(energy_error) == False:
            energy_error_squared_sum = energy_error_squared_sum + count_of_this_layer * energy_error * energy_error
        else:
            print("encountered nan value in energy error, incomplete sum")

    print(1000*time_sum, "[ms]")
    print(1000*math.sqrt(runtime_error_squared_sum), '[ms]')
    print(energy_sum, "[mJ]")
    print(math.sqrt(energy_error_squared_sum), '[mJ]')

