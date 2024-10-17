import os
import shutil

# Define the root folder where your measurements and original datasets are stored
root_folder = "measurements"
original_datasets_folder = os.path.join(root_folder, "original")

def copy_datasets_to_hardware(hardware_name):
    # Path to the hardware folder
    hardware_path = os.path.join(root_folder, hardware_name)
    
    # Ensure the original datasets folder exists
    if not os.path.exists(original_datasets_folder):
        print(f"Error: The original datasets folder '{original_datasets_folder}' does not exist.")
        return
    
    # Create the hardware folder if it doesn't exist
    os.makedirs(hardware_path, exist_ok=True)
    
    # Get the list of all dataset folders from the original datasets directory
    dataset_folders = [f for f in os.listdir(original_datasets_folder) if os.path.isdir(os.path.join(original_datasets_folder, f))]
    
    # Copy each dataset folder to the hardware folder if it doesn't already exist
    for dataset in dataset_folders:
        source_dataset_path = os.path.join(original_datasets_folder, dataset)
        destination_dataset_path = os.path.join(hardware_path, dataset)
        
        if not os.path.exists(destination_dataset_path):
            shutil.copytree(source_dataset_path, destination_dataset_path)
            print(f"Copied dataset '{dataset}' to hardware folder '{hardware_name}'")
        else:
            print(f"Dataset '{dataset}' already exists in hardware folder '{hardware_name}', skipping.")

# Example usage:
copy_datasets_to_hardware("A30_no_tc")

