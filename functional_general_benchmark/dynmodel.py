#### Go here to find model and weights name and capitalization https://pytorch.org/vision/stable/models.html#classification

from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights

args = parse_model_and_weights()

model = get_model_and_weights(args.model, args.weights)
print(f'Loaded model: {args.model} with weights class: {args.weights}')

df = extract_layer_info(model)

# Filter the DataFrame to keep only rows with 'Conv2d' or 'Linear' in the 'Type' column
filtered_df = df[df['Type'].isin(['Conv2d', 'Linear'])]

# Display the filtered DataFrame
print(filtered_df)


##########################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter
import pandas

input_size = (3, 224, 224)

input_data = torch.randn(1, *input_size)

torchinfo_writer = TorchinfoWriter(model,
                                    input_data=input_data,
                                    verbose=0)

torchinfo_writer.construct_model_tree()

df_bigtree = torchinfo_writer.get_dataframe()


# Filter the DataFrame to keep only rows with 'Conv2d' or 'Linear' in the 'Type' column
filtered_df_bigtree = df_bigtree[df_bigtree['Type'].isin(['Conv2d', 'Linear'])]

filtered_df_bigtree = df_bigtree[df_bigtree['Name'].isin([str.contains('fc'), str.contains('conv')])] ## still wrong maybe also think about approach might bo to narrow midned and not rgerenal enough

# filtered_df_bigtree = filtered_df[
#     (filtered_df['Type'] == 'Conv2d') & (filtered_df['Name'].str.contains('conv', case=False)) |
#     (filtered_df['Type'] == 'Linear')
# ]


print(filtered_df_bigtree)

# print(model)

