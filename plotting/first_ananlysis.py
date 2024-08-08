import pandas as pd
from scipy import stats

# Load the CSV file
# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('your_file.csv', delimiter=', ')

# Display the first few rows to understand its structure
print(df.head())

# Replace 'your_column' with the name of the column you want to analyze
column_name = 'your_column'

# Extract the data from the column
data = df[column_name].dropna()  # Drop any NaN values

# Calculate mean
mean = data.mean()

# Calculate standard deviation
std_dev = data.std(ddof=1)  # Use ddof=1 for sample standard deviation

# Calculate median
median = data.median()

print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")
print(f"Median: {median}")

# Optionally, fit a Gaussian distribution and get the parameters
mean_fit, std_dev_fit = stats.norm.fit(data)

print(f"Fitted Mean: {mean_fit}")
print(f"Fitted Standard Deviation: {std_dev_fit}")
