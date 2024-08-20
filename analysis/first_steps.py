import os
import pandas as pd
import numpy as np
import re

# Directory containing the log files
log_dir = 'logs/'

# List to store results for each file
results = []

# Loop through each file in the logs directory
for filename in os.listdir(log_dir):
    if filename.endswith('.log') and not filename.startswith('benchmark'):
        # Extract the number of iterations from the filename using regex
        iterations = int(re.search(r'ifm_(\d+)iter\.log', filename).group(1))
        
        # Load the log file into a pandas DataFrame
        df = pd.read_csv(os.path.join(log_dir, filename), header=None)
        
        # Assign column names
        df.columns = ['Timestamp', 'Value1', 'Value2', 'Value3', 'Value4', 'Category']
        
        # Convert the 'Timestamp' column to datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
        
        # Calculate the time difference between the first and last timestamp in seconds
        time_difference_seconds = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds()
        
        # Exclude the last row for calculations
        df_without_last = df.iloc[:-1]
        
        # Calculate the standard deviation for Value2 before filtering
        std_value2 = df_without_last['Value2'].std()
        
        # Filter out values outside the 3 standard deviation range for Value2
        mean_value2 = df_without_last['Value2'].mean()
        filtered_df = df_without_last[
            (np.abs(df_without_last['Value2'] - mean_value2) <= 3 * std_value2)
        ]
        
        # Calculate the filtered mean and median for Value2
        filtered_mean_value2 = filtered_df['Value2'].mean()
        
        # Calculate the total energy in joules (energy = power * time)
        total_energy_joules = filtered_mean_value2 * time_difference_seconds
        
        # Calculate the energy per iteration
        energy_per_iteration = total_energy_joules / iterations
        
        # Append the results to the list
        results.append({
            'Filename': filename,
            'Iterations': iterations,
            'Time (s)': time_difference_seconds,
            'Filtered Mean (W)': filtered_mean_value2,
            'Standard Deviation (W)': std_value2,
            'Total Energy (J)': total_energy_joules,
            'Energy per Iteration (mJ/iter)': 1000*energy_per_iteration
        })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('energy_calculations.csv', index=False)

print("Energy calculations complete. Results saved to 'energy_calculations.csv'.")
