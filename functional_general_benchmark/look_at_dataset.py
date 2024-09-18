import torch

# Load the saved .pt file
dataset = torch.load('dataset_history/dataset_20240918_145846.pt', map_location=torch.device('cpu'))



for item in dataset:
    print(item)