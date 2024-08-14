import json

# Function to load a JSON file into a list
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to save a list as a JSON file
def save_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Paths to your JSON files
json_file1 = 'print_data_unique.json'
json_file2 = 'tree_data_unique.json'

# Load the JSON files
list1 = load_json_file(json_file1)
list2 = load_json_file(json_file2)

# Merge the lists
merged_list = list1 + list2  # Combine the lists

# Save the merged list to a new JSON file
output_file = 'merged_list.json'
save_json_file(merged_list, output_file)

# Print the merged list of dictionaries
print("Merged List of Dictionaries:")
for item in merged_list:
    print(item)

print(f"\nMerged list saved to {output_file}")
