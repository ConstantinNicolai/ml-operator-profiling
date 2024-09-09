import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.init as init
import pickle
import lzma
from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model, forward_hook_new



with lzma.open("efficientnet_v2_m_3_56_56.pkl.xz") as file_:
    saved_dict = pickle.load(file_)

# print(saved_dict)


filter_list = ['Conv2d','Linear','StochasticDepth']

list_attemps = list(saved_dict.items())


result_list = [entry for entry in list_attemps if entry[0][0] in filter_list]

# print(list_attemps[1][1][0])
# print(list_attemps[1][1][1])
# print(type(list_attemps[1][1][0]))


# for i in range(len(list_attemps)):
#     print(list_attemps[i][1][1])
#     print(list_attemps[i][1][0])
#     print(list_attemps[i][0][2])


for i in range(len(result_list)):
    print(result_list[i][0][0])
    print(result_list[i][0][2])
    print(result_list[i][0][1])
    print(result_list[i])
    # print(list_attemps[i][1][0])
    # print(list_attemps[i][0][2])


