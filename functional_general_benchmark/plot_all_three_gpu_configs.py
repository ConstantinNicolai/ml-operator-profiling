import matplotlib.pyplot as plt
import numpy as np

turquoise_color = 'purple'
maroon_color = 'orange'
fontsize = 11


# Function to read the measurement file
def read_measurement_file(filename):
    measurements = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 5):  # Every 2 lines form a single entry
            model_input_size = lines[i].strip()  # Read the whole line as a key (model + input size)
            energy = float(lines[i+3]) #, float(lines[i+4])
            # energy = float(lines[i+1].split()[2]), float(lines[i+1].split()[6])
            measurements[model_input_size] = energy  # Convert mJ to J
    return measurements

# Function to read the prediction file
def read_prediction_file(filename):
    predictions = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):  # Every 4 lines form a single entry
            model_input_size = lines[i].strip()  # Read the whole line as a key (model + input size)
            energy_mJ = float(lines[i+3].split()[0])  # The energy value (ignore units)
            predictions[model_input_size] = energy_mJ / 1000  # Convert mJ to J
    return predictions

# Example usage
measurement_file = 'A30_fullmodel'
prediction_file = 'A30_no_tc_fullmodel'
third_file = 'RTX2080TI_fullmodel'

measurements = read_measurement_file(measurement_file)
predictions = read_measurement_file(prediction_file)
third_one = read_measurement_file(third_file)

# Ensure the sets of models/input sizes match between measurements and predictions
common_keys = set(measurements.keys()) & set(predictions.keys())

# Prepare data for plotting
models = list(common_keys)
measured_values = [measurements[key]/1000 for key in common_keys]
predicted_values = [predictions[key]/1000 for key in common_keys]
third_values = [third_one[key]/1000 for key in common_keys]
# Set threshold for splitting the y-axis (you can adjust this based on your data)
threshold = 5  # Now in Joules (since mJ to J conversion is done)

# Separate data into two groups: "small" and "large" values
small_indices = [i for i, val in enumerate(measured_values) if val < threshold]
large_indices = [i for i, val in enumerate(measured_values) if val >= threshold]

# Create the grouped bar plot for small values
if small_indices:
    small_models = [models[i] for i in small_indices]
    small_measured = [measured_values[i] for i in small_indices]
    small_predicted = [predicted_values[i] for i in small_indices]
    small_third = [third_values[i] for i in small_indices]

    index = np.arange(len(small_models))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(6, 6))
    bar1 = ax.bar(index, small_measured, bar_width, label='A30 with TC', color=turquoise_color)
    bar2 = ax.bar(index + bar_width, small_predicted, bar_width, label='A30 no TC', color=maroon_color)
    bar3 = ax.bar(index + 2*bar_width, small_third, bar_width, label='2080TI', color="darkgreen")

    # Add labels and titles
    ax.set_xlabel('Model and Input Size')
    ax.set_ylabel('Energy Consumption (J)')  # Updated to Joules
    ax.set_title(f'Measured Energy with and without Tensor Cores A30', fontsize = fontsize)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(small_models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/tc_compare/all_three_small.png', format='png')
    plt.savefig('plots/tc_compare/all_three_small.pdf', format='pdf')

# Create the grouped bar plot for large values
if large_indices:
    large_models = [models[i] for i in large_indices]
    large_measured = [measured_values[i] for i in large_indices]
    large_predicted = [predicted_values[i] for i in large_indices]
    large_third = [third_values[i] for i in large_indices]

    index = np.arange(len(large_models))
    bar_width = 0.25  # Reduced bar width to fit three bars side-by-side

    fig, ax = plt.subplots(figsize=(6, 6))
    bar1 = ax.bar(index, large_measured, bar_width, label='A30 with TC', color=turquoise_color)
    bar2 = ax.bar(index + bar_width, large_predicted, bar_width, label='A30 no TC', color=maroon_color)
    bar3 = ax.bar(index + 2*bar_width, large_third, bar_width, label='2080TI', color="darkgreen")

    # Add labels and titles
    ax.set_xlabel('Model and Input Size')
    ax.set_ylabel('Energy Consumption (J)')
    ax.set_title(f'Measured Energy with and without Tensor Cores A30', fontsize=fontsize)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(large_models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/tc_compare/all_three_large.png', format='png')
    plt.savefig('plots/tc_compare/all_three_large.pdf', format='pdf')
