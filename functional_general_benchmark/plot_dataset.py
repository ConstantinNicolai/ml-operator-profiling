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

# Load the saved .pt file
dataset = torch.load('dataset_history_A30/dataset_20241009_053805.pt', map_location=torch.device('cpu'))  #dataset_20240926_075625.pt
gpu = "A30"


dataset_list = [list(item) for item in dataset]

#if we want to be able to plot anything right here, we need to lower the problems dimensionality


# print(dir(dataset_list[345][0]))
# print(type(dataset_list[345][0].out_channels))

conv2d_list = []
linear_list = []
stochasticdepth_list = []
batchnorm2d_list = []
relu_list = []
adaptiveavgpool2d_list = []




for item in dataset_list:
    if item[0]._get_name() == "Conv2d":
        conv2d_list.append(item)
    elif item[0]._get_name() == "Linear":
        linear_list.append(item)
    elif item[0]._get_name() == "StochasticDepth":
        stochasticdepth_list.append(item)
    elif item[0]._get_name() == "BatchNorm2d":
        batchnorm2d_list.append(item)
    elif item[0]._get_name() == "ReLU":
        relu_list.append(item)
    elif item[0]._get_name() == "AdaptiveAvgPool2d":
        adaptiveavgpool2d_list.append(item)
    else:
        print(print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1]))
        # print("MACs = ",item[0].out_channels*item[1][1]*item[1][2]*item[1][3]*item[0].kernel_size[0]*item[0].kernel_size[1])
        # print(item[0].kernel_size[0],item[0].kernel_size[1])
        # print(" yes, yes , yes !")
    # print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1])
    # print(item)


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
        item3_errors.append(item[4])


# Define the upper limits for x and y
y_limit = 20  # Example: 1,000 for Item[3]

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

# Plot Conv2D with RANSAC quadratic fit and error bars
plt.figure(figsize=(20, 10), dpi=150)
plt.errorbar(macs_conv2d, item3_conv2d, yerr=item3_errors, fmt='o', 
             markersize=8, label='Conv2D Energy in mJ', color='blue', alpha=0.7, 
             capsize=5)  # Adding error bars
plt.plot(macs_range, ransac_fit_line, color='red', label='RANSAC Quadratic Fit', linewidth=1)  # Adding RANSAC quadratic fit line
plt.title('Conv2D Energy vs MACs (Quadratic Fit with Error Bars)' + gpu)
plt.xlabel('MACs (Log Scale)')
plt.ylabel('Energy in mJ')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.ylim(0, y_limit)  # Set the upper limit for y-axis
plt.grid()
plt.legend()
plt.savefig('plots/conv2d_energy_vs_macs_ransac_quadratic_fit_with_errors_' + gpu + '.pdf', format='pdf')
plt.close()

# # Plot Conv2D
# plt.figure(figsize=(20, 10), dpi = 150)
# plt.scatter(macs_conv2d, item3_conv2d, marker='o',s=14, label='Conv2D Item[3]', color='blue')
# plt.title('Conv2D Energy vs MACs (Linear-Log Scale)'+ gpu)
# plt.xlabel('MACs (Log Scale)')
# plt.ylabel('Energy in mJ')
# plt.xscale('log')  # Set x-axis to logarithmic scale
# # plt.yscale('log')  # Set y-axis to logarithmic scale
# plt.ylim(0, y_limit)  # Set the upper limit for y-axis
# plt.grid()
# plt.legend()
# plt.savefig('plots/conv2d_energy_vs_macs_lin_log'+"_"+ gpu +'.png')
# plt.close()

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
plt.savefig('plots/linear_item3_vs_macs_loglog'+"_"+ gpu +'.png')
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
plt.savefig('plots/linear_item3_vs_macs_loglog_zoomin'+"_"+ gpu +'.png')
plt.close()



bn2_energy = []
bn2_cxwxh = []
for item in batchnorm2d_list:
    bn2_energy.append(item[3])
    bn2_cxwxh.append(2*item[1][1]+4*item[1][1]*item[1][2]*item[1][3])


# Plot the filtered data
plt.figure(figsize=(20, 10), dpi=150)
plt.scatter(bn2_cxwxh, bn2_energy, marker='o', s=14, label='BatchNorm2D Energe CxHxW', color='purple')
plt.title('BatchNorm2D FLOPs Energy')
plt.xlabel('FLOPs (Log Scale)')
plt.ylabel('Energy [mJ]')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.yscale('log')  # Set y-axis to logarithmic scale if needed
plt.grid()
plt.legend()
plt.savefig('plots/batchnorm2d_energy_FLOPs'+"_"+ gpu +'.png')
plt.close()




relu_energy = []
relu_cxwxh = []
for item in relu_list:
    relu_energy.append(item[3])
    if len(item[1])<4:
        relu_cxwxh.append(item[1][1])
    else:
        relu_cxwxh.append(item[1][1]*item[1][2]*item[1][3])


# Plot the filtered data
plt.figure(figsize=(20, 10), dpi=150)
plt.scatter(relu_cxwxh, relu_energy, marker='o', s=14, label='ReLU Energe FLOPs', color='limegreen')
plt.title('RELU FLOPs Energy')
plt.xlabel('FLOPs (Log Scale)')
plt.ylabel('Energy [mJ]')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.yscale('log')  # Set y-axis to logarithmic scale if needed
plt.grid()
plt.legend()
plt.savefig('plots/relu_energy_FLOPs'+"_"+ gpu +'.png')
plt.close()





adavpool2d_energy = []
adavpool2d_flops = []
for item in adaptiveavgpool2d_list:
    adavpool2d_energy.append(item[3])
    if isinstance(item[0].output_size, int):
        adavpool2d_flops.append(item[0].output_size*item[1][1]*item[1][2]*item[1][3])
    else:
        adavpool2d_flops.append(item[0].output_size[0]*item[0].output_size[1]*item[1][1]*item[1][2]*item[1][3])


# Plot the filtered data
plt.figure(figsize=(20, 10), dpi=150)
plt.scatter(relu_cxwxh, relu_energy, marker='o', s=14, label='adaptiveavgpool2d Energe FLOPs', color='red')
plt.title('adaptiveavgpool2d FLOPs Energy')
plt.xlabel('FLOPs (Log Scale)')
plt.ylabel('Energy [mJ]')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.grid()
plt.legend()
plt.savefig('plots/adaptiveavgpool2d_energy_FLOPs'+"_"+ gpu +'.png')
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





# Sample list
my_list = [20, 3,1,1, 6]

# Slice the list to exclude the first and last elements
sliced_list = my_list[1:-1]  # This gives [2, 3, 4, 5]

# Use reduce to multiply all elements together
result = reduce(lambda x, y: x * y, sliced_list)

# Print the resulting product
print(result)

print(reduce(lambda x, y: x * y, my_list[1:-1]))