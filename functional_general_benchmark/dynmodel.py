#### Go here to find model and weights name and capitalization https://pytorch.org/vision/stable/models.html#classification

from utils import get_model_and_weights, extract_layer_info, parse_model_and_weights, process_model


# Example usage:
df_counts = process_model(input_size=(3, 224, 224), filter_types=['Conv2d', 'Linear'], exclude_string='downsample')

print(df_counts)