import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.init as init
import pickle
import lzma
import yaml
import os
import glob
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new


filter_list = ['Conv2d','Linear','StochasticDepth', 'BatchNorm2d', 'AvgPool2d', 'AdaptiveAvgPool2d', 'ReLU'] #'Conv2d','Linear','StochasticDepth' , 'ConvTranspose2d'


for summary_file in glob.glob('./../measurements/*/*/summary.yml'):
    HW_dir = os.path.dirname(summary_file)
    with open(summary_file, 'r') as file:
        config = yaml.safe_load(file)


# for entry in os.listdir('./../measurements/*'):
#     with open('./../measurements/*/' + entry + '/summary.yml', 'r') as file:
#         config = yaml.safe_load(file)

    config['input_size'] = tuple(config['input_size'])
        
    tuple_str = "_".join(map(str, config['input_size']))
    filename = f"{config['model_name']}_{tuple_str}.pkl.xz"


    if config['done'] == True:
        print("done flag already set to true, reset to false for rerun")
    if config['done'] == False:

        with lzma.open(HW_dir + '/' + filename) as file_:
            saved_dict = pickle.load(file_)
        

        list_attemps = list(saved_dict.items())

        result_list = [entry for entry in list_attemps if entry[0][0] in filter_list]


        print(HW_dir + '/' + filename)

        with lzma.open(HW_dir + '/' + filename + '_filtered', "wb") as file_:
            pickle.dump(dict(result_list), file_)