import json

# Function to load a JSON file into a list of dictionaries
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to save a list of dictionaries as a JSON file
def save_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Paths to your JSON files
json_file1 = 'print_data_unique.json'
json_file2 = 'tree_data_unique.json'

# Load the JSON files
list1 = load_json_file(json_file1)
list2 = load_json_file(json_file2)

# Check if the lengths match
if len(list1) != len(list2):
    raise ValueError("The lists have different lengths and cannot be merged element-wise.")

# Merge corresponding dictionaries and create a list with only the first dictionary of each pair
merged_list = []
first_entry_list = []

for dict1, dict2 in zip(list1, list2):
    # Merge the two dictionaries
    merged_dict = {**dict1, **dict2}
    merged_list.append(merged_dict)
    
    # Store only the first dictionary in the tuple
    first_entry_list.append(dict1)

# Save the merged list of dictionaries to a new JSON file
output_file_merged = 'merged_list.json'
save_json_file(merged_list, output_file_merged)

# Save the list with only the first entries to a new JSON file
output_file_first_entries = 'first_entry_list.json'
save_json_file(first_entry_list, output_file_first_entries)

# Print the merged list of dictionaries
print("Merged List of Dictionaries:")
for item in merged_list:
    print(item)

print(f"\nMerged list saved to {output_file_merged}")

# Print the list with only the first entries
print("\nList with Only the First Entries of Each Pair:")
for item in first_entry_list:
    print(item)

print(f"\nFirst entries list saved to {output_file_first_entries}")
