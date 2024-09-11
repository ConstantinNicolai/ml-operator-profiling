import pandas as pd
import numpy as np



def process_log_file(in_file, iterations):
    # Load the log file into a pandas DataFrame
    df = pd.read_csv(in_file, header=None)

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

    # Calculate the filtered mean for Value2
    filtered_mean_value2 = filtered_df['Value2'].mean()

    # Calculate the total energy in joules (energy = power * time)
    total_energy_joules = filtered_mean_value2 * time_difference_seconds

    # Calculate the energy per iteration
    energy_per_iteration = total_energy_joules / iterations

    energy_per_iteration_in_milli_joule = 1000 * energy_per_iteration

    # Return the values directly
    return iterations, time_difference_seconds, filtered_mean_wattage, std_wattage, total_energy_joules, energy_per_iteration_in_milli_joule