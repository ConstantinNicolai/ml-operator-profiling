import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.init as init
import pickle
import lzma
import yaml
import os
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new


filter_list = ['Conv2d','Linear','StochasticDepth', 'BatchNorm2d', 'AvgPool2d', 'AdaptiveAvgPool2d', 'ReLU', 'ConvTranspose2d'] #'Conv2d','Linear','StochasticDepth'

for entry in os.listdir('./../measurements'):
    with open('./../measurements/*/' + entry + '/summary.yml', 'r') as file:
        config = yaml.safe_load(file)

    config['input_size'] = tuple(config['input_size'])

    # Dynamically create variables
    for key, value in config.items():
        globals()[key] = value
        
    tuple_str = "_".join(map(str, input_size))
    filename = f"{model_name}_{tuple_str}.pkl.xz"

    if done == False:
        with lzma.open('./../measurements/' + entry + '/' + filename) as file_:
            saved_dict = pickle.load(file_)
        

        list_attemps = list(saved_dict.items())

        result_list = [entry for entry in list_attemps if entry[0][0] in filter_list]


        with lzma.open('./../measurements/*/' + entry + '/' + filename + '_filtered', "wb") as file_:
            pickle.dump(dict(result_list), file_)

