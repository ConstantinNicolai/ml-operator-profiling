import re
import json

def extract_values_from_text(text, default_stride=(1, 1), default_padding=(0, 0)):
    # Regular expression pattern to capture the function name and parameters
    pattern = re.compile(
        r"(\w+)\("
        r"(?:\s*(\d+)\s*,\s*)?"  # Optional in_channels
        r"(\d+)\s*,\s*"  # out_channels
        r"(?:kernel_size=\((\d+),\s*(\d+)\)\s*,\s*)?"  # Optional kernel_size
        r"(?:stride=\((\d+),\s*(\d+)\)\s*,\s*)?"  # Optional stride
        r"(?:padding=\((\d+),\s*(\d+)\)\s*,\s*)?"  # Optional padding
        r"(?:bias=False\))?"  # Optional bias
    )
    
    match = pattern.search(text)
    
    if match:
        # Extract values from the match object
        operator = match.group(1)  # Function name
        in_channels = int(match.group(2)) if match.group(2) else None
        out_channels = int(match.group(3)) if match.group(3) else None
        
        # Extract kernel_size
        kernel_size = (
            (int(match.group(4)), int(match.group(5)))
            if match.group(4) and match.group(5)
            else (None, None)
        )
        
        # Extract stride
        stride = (
            (int(match.group(6)), int(match.group(7)))
            if match.group(6) and match.group(7)
            else default_stride
        )
        
        # Extract padding
        padding = (
            (int(match.group(8)), int(match.group(9)))
            if match.group(8) and match.group(9)
            else default_padding
        )
        
        return {
            "operator": operator,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
    else:
        raise ValueError("The text does not match the expected pattern")

def process_file(filename):
    results = []
    
    # Read the text from the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Process each line and extract values
    for line in lines:
        line = line.strip()
        try:
            result = extract_values_from_text(line)
            results.append(result)
        except ValueError as e:
            print(f"Skipping line due to error: {e}")
    
    return results

def get_unique_data(data):
    # Convert dictionaries to JSON strings to use as set elements
    seen = set()
    unique_data = []
    
    for item in data:
        # Convert the dictionary to a JSON string for comparison
        item_str = json.dumps(item, sort_keys=True)
        if item_str not in seen:
            seen.add(item_str)
            unique_data.append(item)
    
    return unique_data

# Example usage
filename = 'your_file.txt'
data = process_file(filename)

# Get unique lines
data_unique = get_unique_data(data)

# Print the results
print("All data:")
for i, entry in enumerate(data):
    print(f"Line {i+1}: {entry}")

print("\nUnique data:")
for i, entry in enumerate(data_unique):
    print(f"Unique Line {i+1}: {entry}")

# Optionally write results to files
with open('data.json', 'w') as file:
    json.dump(data, file, indent=4)

with open('data_unique.json', 'w') as file:
    json.dump(data_unique, file, indent=4)
