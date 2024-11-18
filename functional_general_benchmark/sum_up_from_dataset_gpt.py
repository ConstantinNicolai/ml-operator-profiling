import torch
import os
import yaml
import lzma
import pickle
import math

# Constants
T_CUTOFF = 0.00034  # Cutoff time in seconds

# Load the saved .pt file
dataset = torch.load('datasets_newbench/dataset_history_A30/dataset_20241110_182037.pt', map_location=torch.device('cpu'))
dataset_list = [list(item) for item in dataset]

for entry in os.listdir('./../measurements/A30'):
    with open('./../measurements/A30/' + entry + '/summary.yml', 'r') as file:
        config = yaml.safe_load(file)

    config['input_size'] = tuple(config['input_size'])
    tuple_str = "_".join(map(str, config['input_size']))
    filename = f"{config['model_name']}_{tuple_str}.pkl.xz"

    with lzma.open('./../measurements/A30/' + entry + '/' + filename + '_filtered') as file_:
        saved_dict = pickle.load(file_)
    
    list_attempts = list(saved_dict.items())
    working_list = [item[1] + [item[0][2]] for item in list_attempts]


    # Add missing entries from the master list to the list to modify, while preserving existing ones
    for item in working_list:
        for master_item in dataset_list:
            if item[0]._get_name() == master_item[0]._get_name() and item[0].extra_repr() == master_item[0].extra_repr() and item[2] == master_item[1] and len(item) == 3:    # module._get_name(), module.extra_repr()
                # Check which items from master_item are missing in item and extend it
                o = 0
                for entry in master_item[1:]:
                    o = o +1
                    if o != 1:
                        item.append(entry)

    # Initialize sums
    time_sum = 0
    energy_sum = 0
    energy_error_squared_sum = 0
    time_error_squared_sum = 0

    for item in working_list:
        count_of_this_layer = item[1]
        runtime = item[3]
        energy = item[4]
        iterations = item[7]
        runtime_for_all_iterations = item[8]
        energy_error = item[6]

        if not math.isnan(runtime):
            # Calculate runtime per iteration and its error
            runtime_per_iteration = runtime_for_all_iterations / iterations
            relative_uncertainty = T_CUTOFF / runtime_for_all_iterations
            runtime_error_per_iteration = runtime_per_iteration * relative_uncertainty

            # Total time for this layer and its contribution to the total time uncertainty
            layer_time = count_of_this_layer * runtime_per_iteration
            layer_time_error = count_of_this_layer * runtime_error_per_iteration

            # Add to the total time and time uncertainty sums
            time_sum += layer_time
            time_error_squared_sum += layer_time_error ** 2
        else:
            print("Encountered NaN value in runtime, incomplete sum")

        if not math.isnan(energy):
            # Sum energy and energy error as before
            energy_sum += count_of_this_layer * energy
            if not math.isnan(energy_error):
                energy_error_squared_sum += (count_of_this_layer * energy_error) ** 2
        else:
            print("Encountered NaN value in energy, incomplete sum")

    # Calculate final uncertainties
    total_time_error = math.sqrt(time_error_squared_sum)
    total_energy_error = math.sqrt(energy_error_squared_sum)

    # Display results
    print(config['model_name'], config['input_size'])


    print(1000*time_sum, "[ms]")
    print(1000 * total_time_error, "[ms]")
    print(energy_sum, "[mJ]")
    print(total_energy_error, '[mJ]')
