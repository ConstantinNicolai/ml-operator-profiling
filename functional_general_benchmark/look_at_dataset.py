import torch
import os
import yaml
import lzma
import pickle
import math

# Load the saved .pt file
dataset = torch.load('dataset_history/dataset_20240918_145846.pt', map_location=torch.device('cpu'))

print("#########################################")

print("type of dataset", type(dataset))
print("tye of dataset entries", type(dataset[0]))




dataset_list = [list(item) for item in dataset]

print("type of datase_list", type(dataset_list))
print("tye of dataset list entries", type(dataset_list[0]))
print(dataset_list[0])



for entry in os.listdir('./../measurements'):
    with open('./../measurements/' + entry + '/summary.yml', 'r') as file:
        config = yaml.safe_load(file)

    config['input_size'] = tuple(config['input_size'])

    # Dynamically create variables
    for key, value in config.items():
        globals()[key] = value
        
    tuple_str = "_".join(map(str, input_size))
    filename = f"{model_name}_{tuple_str}.pkl.xz"


    with lzma.open('./../measurements/' + entry + '/' + filename + '_filtered') as file_:
        saved_dict = pickle.load(file_)
    

    list_attemps = list(saved_dict.items())

    # print("type of list attempts", type(list_attemps))

    # print("type of list attempts antries", type(list_attemps[0]))

    # working_list = []


    print(model_name)

    # print(list_attemps[0][1][0])
    # print(list_attemps[0][1][1])
    # print(len(list_attemps[0]))
    # print(type(list_attemps[0][1][0]))
    # print(type(list_attemps[0][1]))
    # print(list_attemps[0][0][2])


    #working_list =  [item[1] for item in list_attemps] #list_attemps[:][1]  #
    working_list = [item[1] + [item[0][2]] for item in list_attemps]
    

    # print(working_list[0])
    # print(type(working_list))
    # print(type(working_list[0]))

    # print(working_list[0])

    # print("len of working list", len(working_list[0]))


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


    # print(working_list[0])

    time_sum = 0
    energy_sum = 0

    for item in working_list:
        # print(item)
        # print(item[1])
        count_of_this_layer = item[1]
        runtime = item[3]
        energy = item[4]
        if math.isnan(runtime) == False:
            time_sum = time_sum + count_of_this_layer * runtime
        else:
            print("encountered nan value in runtime, incomplete sum")
        if math.isnan(energy) == False:
            energy_sum = energy_sum + count_of_this_layer * energy
        else:
            print("encountered nan value in energy, incomplete sum")

    print(time_sum)
    print(energy_sum)

