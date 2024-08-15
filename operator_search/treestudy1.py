import re
import json
from collections import defaultdict
import sys


# Check if a command-line argument was provided
if len(sys.argv) != 2:
    print("Usage: python example.py <filepath>")
    sys.exit(1)

# Read the command-line argument
path = sys.argv[1]



def extract_values_from_line(line):
    # Regular expression pattern to capture the function name and the last two dimensions of input size and output size
    pattern = re.compile(
        r"(\w+)\s*\[Input Size=\[([0-9]+),\s*[0-9]+,\s*([0-9]+),\s*([0-9]+)\],\s*"
        r"Output Size=\[([0-9]+),\s*[0-9]+,\s*([0-9]+),\s*([0-9]+)\]\]"
    )
    
    match = pattern.search(line)
    
    if match:
        function_name = match.group(1)  # Extract function name
        input_size_last_two = (
            int(match.group(3)),  # Height of input size
            int(match.group(4))   # Width of input size
        )
        output_size_last_two = (
            int(match.group(6)),  # Height of output size
            int(match.group(7))   # Width of output size
        )
        
        return {
            "input_size": input_size_last_two,
            "output_size": output_size_last_two
        }
    else:
        raise ValueError("The line does not match the expected pattern")

def process_file(filename):
    results = []
    
    # Read the text from the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Process each line and extract values
    for line in lines:
        line = line.strip()
        try:
            result = extract_values_from_line(line)
            results.append(result)
        except ValueError as e:
            print(f"Skipping line due to error: {e}")
    
    return results

def get_unique_data(data):
    # Dictionary to hold counts of each unique entry
    count_dict = defaultdict(int)
    unique_data = []
    
    for item in data:
        # Convert the dictionary to a JSON string for comparison
        item_str = json.dumps(item, sort_keys=True)
        count_dict[item_str] += 1
    
    # Convert back from JSON string to dictionary and add count to the unique data
    for item_str, count in count_dict.items():
        item = json.loads(item_str)
        item['count'] = count
        unique_data.append(item)
    
    return unique_data

# Example usage
filename = path
data = process_file(filename)

# Get unique lines with counts
data_unique = get_unique_data(data)

# # Print the results
# print("All data:")
# for i, entry in enumerate(data):
#     print(f"Line {i+1}: {entry}")

# print("\nUnique data with counts:")
# for i, entry in enumerate(data_unique):
#     print(f"Unique Line {i+1}: {entry}")

# # Optionally write results to files
# with open('tree_data.json', 'w') as file:
#     json.dump(data, file, indent=4)

with open(f"{path}.json", 'w') as file:
    json.dump(data_unique, file, indent=4)
