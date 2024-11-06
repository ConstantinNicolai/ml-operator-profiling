import pandas as pd
import matplotlib.pyplot as plt

# Step 4: Set up the plot
plt.figure(figsize=(30, 12))


for i in range(5):
    # Read in the CSV file and set up the path
    log = f"current_temp_RTX2080TI_{i}.log"
    df = pd.read_csv(log, delimiter=',', on_bad_lines='skip', header=None)

    # Assign column names
    df.columns = ['Timestamp', 'Value1', 'Value2', 'Value3', 'Value4']
    
    # Drop any rows that contain NaN values and strip whitespace from strings
    df = df.dropna()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Convert the 'Timestamp' column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    # Convert the other columns to numeric, coercing errors into NaNs, then drop NaNs
    df[['Value1', 'Value2', 'Value3', 'Value4']] = df[['Value1', 'Value2', 'Value3', 'Value4']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Set relative time by subtracting the first timestamp from all timestamps
    df['RelativeTime'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

    # Extract the second value column ('Value2') for plotting
    values = df['Value2']

    # Plot with RelativeTime on the x-axis to align the start times
    plt.scatter(df['RelativeTime'], values, s=0.45, label=f'Power {i}')

# Final plot adjustments
plt.xlabel('Time (seconds)')
plt.ylabel('Power [W]')
plt.title('Aligned Power Usage Over Time')
plt.legend()
plt.grid(True)
plt.savefig('plots/logs/current.png', format='png')
