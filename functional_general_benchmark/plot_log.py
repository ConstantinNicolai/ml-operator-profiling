import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Set up the plot size
plt.figure(figsize=(6,6))

# Loop over files (modify the range or filenames as needed)
for i in range(1):
    # Set up the filename for each iteration
    log = f"current_continous.log"
    df = pd.read_csv(log, delimiter=',', on_bad_lines='skip', header=None)

    # Assign column names
    df.columns = ['Timestamp', 'Value1', 'Value2', 'Value3', 'Value4']
    
    # Clean data by dropping NaN values and stripping whitespace
    df = df.dropna()
    df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)  # Replaced deprecated applymap

    # Convert 'Timestamp' to datetime and drop invalid dates
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    # Convert other columns to numeric and drop rows with NaN values
    df[['Value1', 'Value2', 'Value3', 'Value4']] = df[['Value1', 'Value2', 'Value3', 'Value4']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Create a RelativeTime column to align the start times
    df['RelativeTime'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

    # Extract the second value column ('Value2') for plotting
    y = df['Value2']
    x = df['RelativeTime']

    # Define the threshold to distinguish "low" and "high" states
    threshold = 120  # Adjust this based on data characteristics

    # Identify "high" and "low" states based on the threshold
    high_state = y > threshold

    # Initialize a list to store intervals and detect transitions
    intervals = []
    current_state = high_state.iloc[0]
    start = 0

    # Loop through the data to identify intervals
    for j in range(1, len(high_state)):
        if high_state.iloc[j] != current_state:
            end = j
            intervals.append((start, end, 'high' if current_state else 'low'))
            start = j
            current_state = high_state.iloc[j]

    # Append the last interval
    intervals.append((start, len(high_state) - 1, 'high' if current_state else 'low'))

    # Plot the raw data and highlight "high" and "low" intervals

    plt.axhline(threshold, color='purple', linestyle='--', label='Threshold' if i == 0 else "")

    # Correctly plot intervals within bounds and annotate with duration
    for start, end, state in intervals:
        if start < len(x) and end < len(x):  # Ensure start and end are within bounds
            # Highlight the interval in green (high) or blue (low)
            plt.axvspan(x.iloc[start], x.iloc[end], color='green' if state == 'high' else 'blue', alpha=0.3)
            
            # Calculate the duration of the interval
            duration = x.iloc[end] - x.iloc[start]
            # Annotate the plot with the duration
            # plt.text((x.iloc[start] + x.iloc[end]) / 2, max(y), f'{state}: {duration:.2f}s', 
            #          horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')

    plt.scatter(df['RelativeTime'],df['Value2'] , s = 0.5, label=f'Power {i}', alpha=0.7, color = "red")

# Final plot adjustments
plt.xlabel('Time (seconds)')
plt.ylabel('Power [W]')
plt.title('Continous Power')
plt.legend()
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.7)
plt.grid(which='minor', linestyle=':', linewidth=0.5)

# Save the plot
plt.savefig('plots/logs/current_continous_log.png', format='png')
plt.savefig('plots/logs/current_continous_log.pdf', format='pdf')
