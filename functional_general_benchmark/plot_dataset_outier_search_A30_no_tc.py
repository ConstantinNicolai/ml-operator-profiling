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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


turquoise_color = '#25be48'
maroon_color = '#be259b'

# Load the saved .pt file
dataset = torch.load('datasets/dataset_history_A30_no_tc/dataset_20241023_094928.pt', map_location=torch.device('cpu'))  #dataset_20240926_075625.pt
gpu = "A30_no_tc"
gpu_title = 'A30 no TC'

dataset_list = [list(item) for item in dataset]

#if we want to be able to plot anything right here, we need to lower the problems dimensionality

outliers_reran = torch.load('datasets/outliers_A30_no_tc/dataset_20241104_164724.pt', map_location=torch.device('cpu'))

outliers_list = [list(item) for item in outliers_reran]

conv2d_list = []
linear_list = []
stochasticdepth_list = []
batchnorm2d_list = []
relu_list = []
adaptiveavgpool2d_list = []

outlier_list = []


for item in dataset_list:
    if item[0]._get_name() == "Conv2d":
        if (np.abs(item[3]) < 1000):
            conv2d_list.append(item)
    elif item[0]._get_name() == "Linear":
        linear_list.append(item)
        # print(item[1])
    elif item[0]._get_name() == "StochasticDepth":
        stochasticdepth_list.append(item)
    elif item[0]._get_name() == "BatchNorm2d":
        if (np.abs(item[3]) < 1000):
            batchnorm2d_list.append(item)
    elif item[0]._get_name() == "ReLU":
        relu_list.append(item)
        # print(item)
    elif item[0]._get_name() == "AdaptiveAvgPool2d":
        adaptiveavgpool2d_list.append(item)
    # else:
    #     print(print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1]))
    #     print(item)
        # print("MACs = ",item[0].out_channels*item[1][1]*item[1][2]*item[1][3]*item[0].kernel_size[0]*item[0].kernel_size[1])
        # print(item[0].kernel_size[0],item[0].kernel_size[1])
        # print(" yes, yes , yes !")
    # print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
    # print(item)

for item in outliers_list:
    if item[0]._get_name() == "Conv2d":
        conv2d_list.append(item)
    elif item[0]._get_name() == "BatchNorm2d":
        batchnorm2d_list.append(item)


print("conv2d ", len(conv2d_list))
print("linerar ", len(linear_list))
print("bnorm ", len(batchnorm2d_list))
print("relu ", len(relu_list))
print("adapavg ", len(adaptiveavgpool2d_list))



# for item in conv2d_list:
#     print(item)
#     # print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
#     print("MACs = ",item[0].out_channels*item[1][1]*item[1][2]*item[1][3]*item[0].kernel_size[0]*item[0].kernel_size[1])

# for item in linear_list:
#     print(item)
#     # print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
#     # print(item[0].bias is not None)
#     # print(dir(item[0]))
#     if item[0].bias is not None:
#         print("MACs = ", item[0].out_features*item[0].in_features+item[0].out_features)
#     else:
#         print("MACs = ", item[0].out_features*item[0].in_features)


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
item3_errors= [] 
for item in conv2d_list:
    if item[3] > 0:
        item3_errors.append(item[5])


# Define the upper limits for x and y
y_limit = 650  # Example: 1,000 for Item[3]

# Assuming macs_conv2d, item3_conv2d, and item3_errors are numpy arrays or lists
macs_conv2d = np.array(macs_conv2d).reshape(-1, 1)  # Reshaping for sklearn (expects 2D input)
item3_conv2d = np.array(item3_conv2d)
item3_errors = np.array(item3_errors)  # Ensure errors are also in numpy array format


# Create polynomial features for quadratic fit
quadratic = PolynomialFeatures(degree=2)
macs_quad = quadratic.fit_transform(macs_conv2d)

# Initialize RANSAC regressor with a polynomial model
ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=0.5, residual_threshold=2.0)
ransac.fit(macs_quad, item3_conv2d)

# Predict values using the RANSAC model for plotting the fit
ransac_predicted = ransac.predict(macs_quad)

# Create a range of MACs for plotting the fit line
macs_range = np.linspace(min(macs_conv2d), max(macs_conv2d), 1000).reshape(-1, 1)
macs_range_quad = quadratic.transform(macs_range)
ransac_fit_line = ransac.predict(macs_range_quad)


conv2d_energy = []
conv2d_cxwxh = []
conv2d_error = []
for item in conv2d_list:
    conv2d_energy.append(np.abs(item[3]))
    conv2d_error.append(np.abs(item[5]))
    if len(item[1])<4:
        conv2d_cxwxh.append(item[1][1])
        if (np.abs(item[3]) > 1000):
            print(item)
            outlier_list.append((item[0],item[1]))
    else:
        conv2d_cxwxh.append(item[1][1]*item[1][2]*item[1][3])
        if (np.abs(item[3]) > 1000):
            print(item)
            outlier_list.append((item[0],item[1]))





# Plot Conv2D with RANSAC quadratic fit and error bars
plt.figure(figsize=(6, 6))
plt.errorbar(conv2d_cxwxh, conv2d_energy, yerr=conv2d_error, fmt='.', 
             markersize=8, label='Conv2D Energy', color='darkblue', alpha=0.7, 
             capsize=5)  # Adding error bars
#plt.plot(macs_range, ransac_fit_line, color='red', label='RANSAC Quadratic Fit', linewidth=1)  # Adding RANSAC quadratic fit line
plt.title('Conv2D Energy vs Ifmap Size ' + gpu_title)
plt.xlabel('Input Feature Map Size')
plt.ylabel('Energy Consumption [mJ]')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.ylim(0, y_limit)  # Set the upper limit for y-axis
plt.grid()
plt.legend()
plt.savefig('plots/ifmap/conv2d_energy_vs_ifmap_' + gpu + '.pdf', format='pdf')
plt.savefig('plots/ifmap/conv2d_energy_vs_ifmap_' + gpu + '.png', format='png')
plt.close()

# # Plot Conv2D
# plt.figure(figsize=(20, 10), dpi = 150)
# plt.scatter(macs_conv2d, item3_conv2d, marker='o',s=14, label='Conv2D Item[3]', color='blue')
# plt.title('Conv2D Energy vs MACs (Linear-Log Scale)'+ gpu)
# plt.xlabel('MACs (Log Scale)')
# plt.ylabel('Energy Consumption [mJ]')
# plt.xscale('log')  # Set x-axis to logarithmic scale
# # plt.yscale('log')  # Set y-axis to logarithmic scale
# plt.ylim(0, y_limit)  # Set the upper limit for y-axis
# plt.grid()
# plt.legend()
# plt.savefig('plots/ifmap/conv2d_energy_vs_macs_lin_log'+"_"+ gpu +'.png')
# plt.close()

# Prepare data for Linear
macs_linear, item3_linear = prepare_data(linear_list)

item3_errors= [] 
for item in linear_list:
    if item[3] > 0:
        item3_errors.append(item[5])

item3_errors = np.array(item3_errors)

# Define the upper limits for x and y
# y_limit = 30  # Example: 1,000 for Item[3]




linear_energy = []
linear_cxwxh = []
linear_error = []
for item in linear_list:
    linear_energy.append(np.abs(item[3]))
    linear_error.append(np.abs(item[5]))
    if len(item[1])<4:
        linear_cxwxh.append(item[1][1])
    else:
        linear_cxwxh.append(item[1][1]*item[1][2]*item[1][3])


# Plot Linear
plt.figure(figsize=(6, 6))
plt.errorbar(linear_cxwxh, linear_energy, yerr=linear_error, fmt='.', 
             markersize=8, label='Linear Energy', color='k', alpha=0.7, 
             capsize=5)  # Adding error bars
#plt.scatter(macs_linear, item3_linear, marker='o', s=14, label='Linear Item[3]', color='orange')
plt.title('Linear Energy vs Ifmap Size '+gpu_title)
plt.xlabel('Input Feature Map Size')
plt.ylabel('Energy Consumption [mJ]')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.yscale('log')  # Set y-axis to logarithmic scale
# plt.ylim(0, y_limit)  # Set the upper limit for y-axis
plt.grid()
plt.legend()
plt.savefig('plots/ifmap/linear_energy_vs_ifmap'+"_"+ gpu +'.png')
plt.savefig('plots/ifmap/linear_energy_vs_ifmap'+"_"+ gpu +'.pdf', format = 'pdf')
plt.close()



# # Define the upper limits for x and y
# # x_limit = 5e6  # Example: 1,000,000 MACs
# # y_limit = 20  # Example: 1,000 for Item[3]

# # Filter the data to only include points within the specified limits
# filtered_macs = [x for x, y in zip(macs_linear, item3_linear)] #if x <= x_limit and y <= y_limit]
# filtered_item3 = [y for x, y in zip(macs_linear, item3_linear)] #if x <= x_limit and y <= y_limit]

# # Plot the filtered data
# plt.figure(figsize=(20, 10), dpi=150)
# plt.scatter(filtered_macs, filtered_item3, marker='o', s=14, label='Linear Item[3]', color='orange')
# plt.title('Linear Energy vs MACs (Log-Log Scale)')
# plt.xlabel('MACs (Log Scale)')
# plt.ylabel('Item[3] (Log Scale)')
# plt.xscale('log')  # Set x-axis to logarithmic scale
# # plt.yscale('log')  # Set y-axis to logarithmic scale if needed
# # plt.xlim(None, x_limit)  # Set the upper limit for x-axis
# # plt.ylim(None, y_limit)  # Set the upper limit for y-axis
# plt.grid()
# plt.legend()
# plt.savefig('plots/ifmap/linear_item3_vs_macs_loglog_zoomin'+"_"+ gpu +'.png')
# plt.close()



bn2_energy = []
bn2_cxwxh = []
bn2_error = []
for item in batchnorm2d_list:
    bn2_energy.append(np.abs(item[3]))
    bn2_cxwxh.append(item[1][1]*item[1][2]*item[1][3])#bn2_cxwxh.append(2*item[1][1]+4*item[1][1]*item[1][2]*item[1][3])
    bn2_error.append(np.abs(item[5]))
    if (np.abs(item[3]) > 1000):
        print(item)
        outlier_list.append((item[0],item[1]))

#print(batchnorm2d_list[32])

y_limit = 300

# Plot the filtered data
plt.figure(figsize=(6, 6))
plt.errorbar(bn2_cxwxh, bn2_energy, yerr=bn2_error, fmt='.', 
             markersize=8, label='BatchNorm2D FLOPs Energy', color='purple', alpha=0.7, 
             capsize=5)
#plt.scatter(bn2_cxwxh, bn2_energy, marker='o', s=14, label='BatchNorm2D Energy CxHxW', color='purple')
plt.title('BatchNorm2D Energy Ifmap Size '+ gpu_title)
plt.xlabel('Input Feature Map Size')
plt.ylabel('Energy Consumption [mJ]')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.yscale('log')  # Set y-axis to logarithmic scale if needed
# plt.ylim(0, y_limit)
plt.grid()
plt.legend()
plt.savefig('plots/ifmap/batchnorm2d_energy_ifmap'+"_"+ gpu +'.png')
plt.savefig('plots/ifmap/batchnorm2d_energy_ifmap'+"_"+ gpu +'.pdf', format = 'pdf')
plt.close()




relu_energy = []
relu_cxwxh = []
relu_error = []
for item in relu_list:
    relu_energy.append(item[3])
    relu_error.append(np.abs(item[5]))
    if len(item[1])<4:
        relu_cxwxh.append(item[1][1])
    else:
        relu_cxwxh.append(item[1][1]*item[1][2]*item[1][3])


# Plot the filtered data
plt.figure(figsize=(6, 6))
plt.errorbar(relu_cxwxh, relu_energy, yerr=relu_error, fmt='.', 
             markersize=8, label='ReLU Energy Ifmap Size', color='limegreen', alpha=0.7, 
             capsize=5)
#plt.scatter(relu_cxwxh, relu_energy, marker='o', s=14, label='ReLU Energy FLOPs', color='limegreen')
plt.title('RELU Energy Ifmap Size '+ gpu_title)
plt.xlabel('Input Feature Map Size')
plt.ylabel('Energy Consumption [mJ]')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.ylim(0, 150)
# plt.yscale('log')  # Set y-axis to logarithmic scale if needed
plt.grid()
plt.legend()
plt.savefig('plots/ifmap/relu_energy_ifmap'+"_"+ gpu +'.png')
plt.savefig('plots/ifmap/relu_energy_ifmap'+"_"+ gpu +'.pdf', format = 'pdf')
plt.close()





adavpool2d_energy = []
adavpool2d_flops = []
adavpool2d_error = []
for item in adaptiveavgpool2d_list:
    adavpool2d_energy.append(item[3])
    adavpool2d_error.append(np.abs(item[5]))
    # print(item[0].output_size)
    adavpool2d_flops.append(item[1][1]*item[1][2]*item[1][3])
    # if isinstance(item[0].output_size, int):
    #     adavpool2d_flops.append(item[0].output_size*item[1][1]*item[1][2]*item[1][3])
    # else:
    #     adavpool2d_flops.append(item[0].output_size[0]*item[0].output_size[1]*item[1][1]*item[1][2]*item[1][3])


# Plot the filtered data
plt.figure(figsize=(6, 6))
plt.errorbar(adavpool2d_flops, adavpool2d_energy, yerr=adavpool2d_error, fmt='.', 
             markersize=8, label='adaptiveavgpool2d Energy Ifmap Size', color='red', alpha=0.7, 
             capsize=5)
#plt.scatter(relu_cxwxh, relu_energy, marker='o', s=14, label='adaptiveavgpool2d Energy FLOPs', color='red')
plt.title('adaptiveavgpool2d Energy Ifmap Size '+gpu_title)
plt.xlabel('Input Feature Map')
plt.ylabel('Energy Consumption [mJ]')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.ylim(0, 20)
plt.grid()
plt.legend()
plt.savefig('plots/ifmap/adaptiveavgpool2d_energy_ifmap'+"_"+ gpu +'.png')
plt.savefig('plots/ifmap/adaptiveavgpool2d_energy_ifmap'+"_"+ gpu +'.pdf', format = 'pdf')
plt.close()



print("#####################################")

# print(batchnorm2d_list[34])
# print(batchnorm2d_list[34][1])
# print(batchnorm2d_list[34][1][0])
# print(batchnorm2d_list[34][1][1]*batchnorm2d_list[34][1][2]*batchnorm2d_list[34][1][3])

# print(adaptiveavgpool2d_list[2])
# print(adaptiveavgpool2d_list[2][0].output_size)
# print(type(adaptiveavgpool2d_list[2][0].output_size[0]))
# #print(dir(adaptiveavgpool2d_list[4][0]))

# for item in adaptiveavgpool2d_list:
#     print(item[0].output_size)
#     print(type(item[0].output_size))


print("######################################")


# for item in linear_list:
#     if len(item[1])>2:
#         print(item)



with lzma.open('outlier_rerun_'+gpu, "wb") as file_:
            pickle.dump(dict(outlier_list), file_)

# # Sample list
# my_list = [20, 3,1,1, 6]

# # Slice the list to exclude the first and last elements
# sliced_list = my_list[1:-1]  # This gives [2, 3, 4, 5]

# # Use reduce to multiply all elements together
# result = reduce(lambda x, y: x * y, sliced_list)

# # Print the resulting product
# print(result)

# print(reduce(lambda x, y: x * y, my_list[1:-1]))