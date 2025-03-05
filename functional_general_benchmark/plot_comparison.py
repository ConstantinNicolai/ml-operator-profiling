import matplotlib.pyplot as plt
import numpy as np

# Function to read the measurement file
def read_measurement_file(filename):
    measurements = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):  # Every 2 lines form a single entry
            model_input_size = lines[i].strip()  # Read the whole line as a key (model + input size)
            # Split the second line and extract the energy value (ignore units)
            energy_mJ = float(lines[i+2])  # The fourth element is the energy value, ignore units
            measurements[model_input_size] = energy_mJ
    return measurements

# Function to read the prediction file
def read_prediction_file(filename):
    predictions = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):  # Every 4 lines form a single entry
            model_input_size = lines[i].strip()  # Read the whole line as a key (model + input size)
            # Split the third line and extract the energy value (ignore units)
            energy_mJ = float(lines[i+3].split()[0])  # The second element is the energy value, ignore units
            predictions[model_input_size] = energy_mJ
    return predictions

# Example usage
measurement_file = 'datasets_fullmodel_validation/dataset_history_A30/fullmodel.txt'
prediction_file = '../random_forest/model_predictions.txt'


measurements = read_measurement_file(measurement_file)
predictions = read_measurement_file(prediction_file)

# Ensure the sets of models/input sizes match between measurements and predictions
common_keys = set(measurements.keys()) & set(predictions.keys())

# Prepare data for plotting
models = sorted(common_keys)
measured_values = [measurements[key] for key in common_keys]
predicted_values = [predictions[key] for key in common_keys]

# Create the grouped bar plot
bar_width = 0.35
index = np.arange(len(models))

fig, ax = plt.subplots(figsize=(15, 10))
bar1 = ax.bar(index, measured_values, bar_width, label='Measured', color='g')
bar2 = ax.bar(index + bar_width, predicted_values, bar_width, label='Predicted', color='b')

# Add labels and titles
ax.set_xlabel('Model and Input Size')
ax.set_ylabel('Energy Consumption [mJ]')
ax.set_title('Comparison of Measured and Predicted Energy Consumption')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('plots/validation/full_models_compare.png', format='png')
plt.savefig('plots/validation/full_models_compare.pdf', format='pdf')
