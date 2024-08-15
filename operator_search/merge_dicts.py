import json
import sys

# Check if a command-line argument was provided
if len(sys.argv) != 4:
    print("Usage: python example.py <filepath>")
    sys.exit(1)

# Read the command-line argument
path0 = sys.argv[1]
path1 = sys.argv[2]
output = sys.argv[3]

# Function to load a JSON file into a list of dictionaries
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to save a list of dictionaries as a JSON file
def save_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to process the dictionaries to flatten certain fields
def process_dicts(dict_list):
    processed_list = []
    for d in dict_list:
        processed_dict = d.copy()
        
        # Flatten kernel_size, padding, stride, input_size, output_size
        if isinstance(processed_dict.get('kernel_size'), list):
            processed_dict['kernel_size'] = processed_dict['kernel_size'][0]
        
        if isinstance(processed_dict.get('padding'), list):
            processed_dict['padding'] = processed_dict['padding'][0]
        
        if isinstance(processed_dict.get('stride'), list):
            processed_dict['stride'] = processed_dict['stride'][0]

        if isinstance(processed_dict.get('input_size'), list):
            processed_dict['input_size'] = processed_dict['input_size'][0]
        
        if isinstance(processed_dict.get('output_size'), list):
            processed_dict['output_size'] = processed_dict['output_size'][0]

        processed_list.append(processed_dict)
    
    return processed_list

# Function to merge corresponding dictionaries from two lists
def merge_lists(list1, list2):
    max_length = max(len(list1), len(list2))
    merged_list = []
    
    for i in range(max_length):
        dict1 = list1[i] if i < len(list1) else {}
        dict2 = list2[i] if i < len(list2) else {}
        merged_dict = {**dict1, **dict2}
        merged_list.append(merged_dict)
    
    return merged_list

# Paths to your JSON files
json_file1 = path0
json_file2 = path1

list1 = load_json_file(json_file1)
list2 = load_json_file(json_file2)

# Process the lists to flatten the specified fields
processed_list1 = process_dicts(list1)
processed_list2 = process_dicts(list2)

# Merge the lists
merged_list = merge_lists(processed_list1, processed_list2)

# Save the merged list of dictionaries to a new JSON file
output_file_merged = f"{output}_merged.json"
save_json_file(merged_list, output_file_merged)

# Print the merged list of dictionaries
print("Merged List of Dictionaries:")
for item in merged_list:
    print(item)
