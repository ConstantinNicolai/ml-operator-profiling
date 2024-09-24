import torch 

# Initializing a linear layer with 
# 2 independent features and 3 dependent features 
linear_layer = torch.nn.Linear(2, 3).cuda()

# Initializing the weights with the Xavier initialization method 
torch.nn.init.xavier_uniform_(linear_layer.weight) 

# Displaying the initialized weights 
print(linear_layer.weight) 


# Load the saved .pt file
dataset = torch.load('dataset_history/dataset_20240923_133127.pt', map_location=torch.device('cpu'))


dataset_list = [list(item) for item in dataset]



conv2d_list = []
linear_list = []
stochasticdepth_list = []




for item in dataset_list:
    if item[0]._get_name() == "Conv2d":
        conv2d_list.append(item)
    elif item[0]._get_name() == "Linear":
        linear_list.append(item)
    elif item[0]._get_name() == "StochasticDepth":
        stochasticdepth_list.append(item)
    else:
        print(print(item[0]._get_name(), item[0].extra_repr(), type(item[0].extra_repr()), item[1]))



print(linear_list[0][0])

somelist = []


for item in linear_list:
    nana = item[0].cuda()

    torch.nn.init.xavier_uniform_(nana.weight)
    torch.nn.init.uniform_(nana.bias, a=-0.1, b=0.1)

    somelist.append(nana)


# nana = linear_list[0][0].cuda()

# torch.nn.init.xavier_uniform_(nana.weight)
# torch.nn.init.xavier_uniform_(nana.bias)

# print(nana.bias)

print(len(linear_list))
print(len(somelist))
