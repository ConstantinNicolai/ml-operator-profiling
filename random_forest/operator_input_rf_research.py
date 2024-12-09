import torch
import os
import yaml
import lzma
import pickle
import math
import numpy as np

# Load the saved .pt file
dataset = torch.load('../functional_general_benchmark/datasets_finalbench/dataset_history_A30/dataset_20241117_202731.pt', map_location=torch.device('cpu'))


dataset_list = [list(item) for item in dataset]

for k in range(len(dataset_list)):
    if dataset_list[k][0]._get_name() == 'ConvTranspose2d':
        print(dataset_list[k][0]._get_name())
        print(dir(dataset_list[k][0]))
        print(dataset_list[k][0])
        # print(dataset_list[k][0].parameters)
        break
        


print("##########################")

# print(dataset_list[21][0]._get_name())
# print(dataset_list[21][0])

# # print(dataset_list[0][0]._parameters) # These are the weights for conv2d

# print(dataset_list[0][0]._version)

# print(dataset_list[0][0].bias)

# print(dataset_list[0][0].compile())

# # print(dataset_list[0][0].cpu())

# print(dataset_list[0][0].dilation)

# # print(dataset_list[0][0].extra_repr())

# # print(type(dataset_list[0][0].extra_repr()))

# print(dataset_list[0][0].in_channels)

# print(dataset_list[0][0].out_channels)

# print(dataset_list[0][0].kernel_size)

# print(dataset_list[0][1])

# print(dataset_list[0][0].output_padding)

# print(dataset_list[0][0].padding)

# print(dataset_list[0][0].padding_mode)

# # print(dataset_list[0][0].type())

# print(type(np.shape(dataset_list[0][0].weight)[0]))

###################################

# print(dataset_list[22][0].in_features)

# print(dataset_list[22][0].out_features)

# print(np.shape(dataset_list[22][0].weight))

# print(dataset_list[22][0].extra_repr())

# print(dataset_list[22][1])

###########################################

# print(dataset_list[2][0].parameters())

########################################

# print(np.shape(dataset_list[1][0].bias))

# # print(dataset_list[1][0].num_batches_tracked)

# print(dataset_list[1][0].num_features)

# print(dataset_list[1][0].bias is None)

#############################################

# print(dataset_list[21][0].output_size)

# print(dataset_list[21][0].parameters())

################################################


