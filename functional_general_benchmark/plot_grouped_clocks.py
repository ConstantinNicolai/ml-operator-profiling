import matplotlib.pyplot as plt
import numpy as np
import os

# Your function to read a measurement file
def read_measurement_file(filename):
    measurements = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):  # Adjusted to your format
            model_input_size = lines[i].strip()
            energy_mJ = float(lines[i+1])  # You can change this to i+1 or so for runtime
            measurements[model_input_size] = energy_mJ
    return measurements

# Define your clock speeds (in order or sort them)
clock_speeds = ['210', '300','600','900', '1200', '1440']
clock_speeds.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  # Sort numerically

# Base path
base_path = 'datasets_fullmodel_inf_validation/'

# Use a set to collect all models across all clocks
all_models = set()

# Read measurements for all clock speeds
data_by_clock = {}
for clock in clock_speeds:
    file_path = os.path.join(base_path, f'dataset_history_A30_{clock}', 'prediction.txt')
    measurements = read_measurement_file(file_path)
    data_by_clock[clock] = measurements
    all_models.update(measurements.keys())

# Sort models to keep the plot consistent
models = sorted(all_models)

# Create grouped bar plot
bar_width = 0.1
x = np.arange(len(models))
fig, ax = plt.subplots(figsize=(15, 8))

for i, clock in enumerate(clock_speeds):
    # Some models might be missing from some clocks – default to 0
    measurements = [data_by_clock[clock].get(model, 0) for model in models]  # mJ to J
    ax.bar(x + i * bar_width, measurements, bar_width, label=clock)

# Axis and labels
ax.set_xlabel('Model and Input Size')
ax.set_ylabel('Runtime [ms]')  # Change this if you switch to runtime
# ax.set_title('Measured Energy Consumption Across Clock Speeds')
ax.set_xticks(x + (bar_width * (len(clock_speeds) - 1) / 2))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(title='Clock Speed [MHz]')

plt.tight_layout()
plt.savefig('plots/clocks/predicted_time_across_clocks_inference.png', format = 'png')
plt.savefig('plots/clocks/predicted_time_across_clocks_inference.pdf', format = 'pdf')