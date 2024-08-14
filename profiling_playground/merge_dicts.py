import json

# Function to load a JSON file into a dictionary
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to save a dictionary as a JSON file
def save_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to merge two dictionaries
def merge_dicts(dict1, dict2):
    # Merging the two dictionaries
    merged_dict = dict1.copy()  # Start with dict1's keys and values
    merged_dict.update(dict2)   # Update with dict2's keys and values, overwriting any duplicates
    return merged_dict

# Paths to your JSON files
json_file1 = 'print_data_unique.json'
json_file2 = 'print_data_unique.json'

# Load the JSON files
dict1 = load_json_file(json_file1)
dict2 = load_json_file(json_file2)

# Merge the dictionaries
merged_dict = merge_dicts(dict1, dict2)

# Save the merged dictionary to a new JSON file
output_file = 'merged_dict.json'
save_json_file(merged_dict, output_file)

print(f"Merged dictionary saved to {output_file}")
