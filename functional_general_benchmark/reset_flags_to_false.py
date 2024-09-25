import os
import yaml

def update_yml_files(directory):
    # Walk through all files in the given directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Process only .yml files
            if file.endswith(".yml"):
                file_path = os.path.join(root, file)
                
                # Open and load the YAML file
                with open(file_path, 'r') as yml_file:
                    try:
                        data = yaml.safe_load(yml_file)
                    except yaml.YAMLError as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                
                # Check if the 'done' key exists and set it to False
                if isinstance(data, dict) and 'done' in data:
                    data['done'] = False
                
                    # Write the updated data back to the YAML file
                    with open(file_path, 'w') as yml_file:
                        yaml.safe_dump(data, yml_file)
                    
                    print(f"Updated 'done' to False in: {file_path}")
                else:
                    print(f"No 'done' key found in: {file_path}")

# Replace with your directory path
directory_path = "../measurements/A30"
update_yml_files(directory_path)
