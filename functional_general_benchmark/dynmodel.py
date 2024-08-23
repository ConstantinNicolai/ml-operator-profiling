#### Go here to find model and weights name and capitalization https://pytorch.org/vision/stable/models.html#classification



import argparse
import importlib
import torch
from torchvision import models, transforms
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import re
from utils import get_model_and_weights, extract_layer_info

parser = argparse.ArgumentParser(description='Load a model and its weights.')
parser.add_argument('--model', type=str, required=True,
                    help='Name of the model to load (e.g., "resnet50", "vgg16").')
parser.add_argument('--weights', type=str, required=True,
                    help='Name of the weights class to load (e.g., "ResNet50_Weights", "VGG16_Weights").')

args = parser.parse_args()

model = get_model_and_weights(args.model, args.weights)
print(f'Loaded model: {args.model} with weights class: {args.weights}')

df = extract_layer_info(model)

# Filter the DataFrame to keep only rows with 'Conv2d' or 'Linear' in the 'Type' column
filtered_df = df[df['Type'].isin(['Conv2d', 'Linear'])]

# Display the filtered DataFrame
print(filtered_df)