import torch
import os
import yaml
import lzma
import pickle
import math
import matplotlib.pyplot as plt
import os
import numpy as np
from functools import reduce


# Load the saved .pt file
dataset = torch.load('dataset_history/dataset_20240926_075625.pt', map_location=torch.device('cpu'))


dataset_list = [list(item) for item in dataset]

#if we want to be able to plot anything right here, we need to lower the problems dimensionality


# print(dir(dataset_list[345][0]))
# print(type(dataset_list[345][0].out_channels))

conv2d_list = []
linear_list = []
stochasticdepth_list = []
batchnorm2d_list = []




for item in dataset_list:
    if item[0]._get_name() == "Conv2d":
        conv2d_list.append(item)
    elif item[0]._get_name() == "Linear":
        linear_list.append(item)
    elif item[0]._get_name() == "StochasticDepth":
        stochasticdepth_list.append(item)
    elif item[0]._get_name() == "BatchNorm2d":
        batchnorm2d_list.append(item)
    else:
        print(print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1]))
        # print("MACs = ",item[0].out_channels*item[1][1]*item[1][2]*item[1][3]*item[0].kernel_size[0]*item[0].kernel_size[1])
        # print(item[0].kernel_size[0],item[0].kernel_size[1])
        # print(" yes, yes , yes !")
    # print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
    # print(item)


for item in conv2d_list:
    print(item)
    # print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
    print("MACs = ",item[0].out_channels*item[1][1]*item[1][2]*item[1][3]*item[0].kernel_size[0]*item[0].kernel_size[1])

for item in linear_list:
    print(item)
    # print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
    # print(item[0].bias is not None)
    # print(dir(item[0]))
    if item[0].bias is not None:
        print("MACs = ", item[0].out_features*item[0].in_features+item[0].out_features)
    else:
        print("MACs = ", item[0].out_features*item[0].in_features)


# for item in stochasticdepth_list:
#     print(item[0].get_parameter)


# # Create plots directory if it doesn't exist
# os.makedirs('plots', exist_ok=True)

# # Plot Conv2D
# macs_conv2d = [
#     item[0].out_channels * item[1][1] * item[1][2] * item[1][3] * item[0].kernel_size[0] * item[0].kernel_size[1] 
#     for item in conv2d_list
# ]
# item3_conv2d = [item[3] for item in conv2d_list]

# plt.figure(figsize=(10, 5))
# plt.scatter(macs_conv2d, item3_conv2d, marker='o', label='Conv2D Item[3]', color='blue')
# plt.title('Conv2D Item[3] vs MACs (Log Scale)')
# plt.xlabel('MACs (Log Scale)')
# plt.ylabel('Item[3]')
# plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.grid()
# plt.legend()
# plt.savefig('plots/conv2d_item3_vs_macs_log.png')
# plt.close()

# # Plot Linear
# macs_linear = [
#     (item[0].out_features * item[0].in_features + item[0].out_features) if item[0].bias is not None 
#     else (item[0].out_features * item[0].in_features) 
#     for item in linear_list
# ]
# item3_linear = [item[3] for item in linear_list]

# plt.figure(figsize=(10, 5))
# plt.scatter(macs_linear, item3_linear, marker='o', label='Linear Item[3]', color='orange')
# plt.title('Linear Item[3] vs MACs (Log Scale)')
# plt.xlabel('MACs (Log Scale)')
# plt.ylabel('Item[3]')
# plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.grid()
# plt.legend()
# plt.savefig('plots/linear_item3_vs_macs_log.png')
# plt.close()





# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Function to filter and prepare data for plotting
def prepare_data(layers):
    macs = []
    item3 = []
    
    for item in layers:
        if isinstance(item[3], (int, float)):  # Ensure item[3] is a number
            mac = (item[0].out_channels * item[1][1] * item[1][2] * item[1][3] * item[0].kernel_size[0] * item[0].kernel_size[1] 
                    if item[0]._get_name() == "Conv2d" 
                    else ((item[0].out_features * item[0].in_features + item[0].out_features)* (reduce(lambda x, y: x * y, item[1][1:-1]) if len(item[1]) > 2 else 1)) if item[0].bias is not None 
                    else ((item[0].out_features * item[0].in_features))*(reduce(lambda x, y: x * y, item[1][1:-1]) if len(item[1]) > 2 else 1))
            if mac > 0 and item[3] > 0:  # Ensure both mac and item[3] are positive
                macs.append(mac)
                item3.append(item[3])
    
    return macs, item3

# Prepare data for Conv2D
macs_conv2d, item3_conv2d = prepare_data(conv2d_list)

# Plot Conv2D
plt.figure(figsize=(20, 10), dpi = 150)
plt.scatter(macs_conv2d, item3_conv2d, marker='o',s=14, label='Conv2D Item[3]', color='blue')
plt.title('Conv2D Energy vs MACs (Linear-Log Scale)')
plt.xlabel('MACs (Log Scale)')
plt.ylabel('Energy in mJ')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.yscale('log')  # Set y-axis to logarithmic scale
plt.grid()
plt.legend()
plt.savefig('plots/conv2d_energy_vs_macs_lin_log.png')
plt.close()

# Prepare data for Linear
macs_linear, item3_linear = prepare_data(linear_list)


# Plot Linear
plt.figure(figsize=(20, 10), dpi = 150)
plt.scatter(macs_linear, item3_linear, marker='o', s=14, label='Linear Item[3]', color='orange')
plt.title('Linear Energy vs MACs (Log-Log Scale)')
plt.xlabel('MACs (Log Scale)')
plt.ylabel('Item[3] (Log Scale)')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.yscale('log')  # Set y-axis to logarithmic scale
plt.grid()
plt.legend()
plt.savefig('plots/linear_item3_vs_macs_loglog.png')
plt.close()



# Define the upper limits for x and y
x_limit = 5e6  # Example: 1,000,000 MACs
y_limit = 20  # Example: 1,000 for Item[3]

# Filter the data to only include points within the specified limits
filtered_macs = [x for x, y in zip(macs_linear, item3_linear) if x <= x_limit and y <= y_limit]
filtered_item3 = [y for x, y in zip(macs_linear, item3_linear) if x <= x_limit and y <= y_limit]

# Plot the filtered data
plt.figure(figsize=(20, 10), dpi=150)
plt.scatter(filtered_macs, filtered_item3, marker='o', s=14, label='Linear Item[3]', color='orange')
plt.title('Linear Energy vs MACs (Log-Log Scale)')
plt.xlabel('MACs (Log Scale)')
plt.ylabel('Item[3] (Log Scale)')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.yscale('log')  # Set y-axis to logarithmic scale if needed
plt.xlim(None, x_limit)  # Set the upper limit for x-axis
plt.ylim(None, y_limit)  # Set the upper limit for y-axis
plt.grid()
plt.legend()
plt.savefig('plots/linear_item3_vs_macs_loglog_zoomin.png')
plt.close()


# insize = []
# enen = []


# for item in linear_list:
#     insize.append(item[1][1])
#     enen.append(item[3])

# # Plot Linear
# plt.figure(figsize=(20, 10), dpi = 150)
# plt.scatter(insize, enen, marker='o', s=14, label='Linear Item[3]', color='green')
# plt.title('Linear Energy vs insize (Log-Log Scale)')
# plt.xlabel('insize')
# plt.ylabel('Item[3] (Log Scale)')
# # plt.xscale('log')  # Set x-axis to logarithmic scale
# # plt.yscale('log')  # Set y-axis to logarithmic scale
# plt.grid()
# plt.legend()
# plt.savefig('plots/linear_item3_vs_insize_loglog.png')
# plt.close()



print("######################################")


for item in linear_list:
    if len(item[1])>2:
        print(item)





# Sample list
my_list = [20, 3,1,1, 6]

# Slice the list to exclude the first and last elements
sliced_list = my_list[1:-1]  # This gives [2, 3, 4, 5]

# Use reduce to multiply all elements together
result = reduce(lambda x, y: x * y, sliced_list)

# Print the resulting product
print(result)

print(reduce(lambda x, y: x * y, my_list[1:-1]))