import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress



# Load the data from the file
data = np.loadtxt('iter_A30', delimiter=' ')

# Assign columns: First column to iterations, second column to runtime
iterations = data[:, 0]
runtime = data[:, 1]

# Sort data by runtime for better handling of top 6/7 segment
sorted_indices = np.argsort(runtime)
iterations = iterations[sorted_indices]
runtime = runtime[sorted_indices]

# Select the top 6/7 of the data
top_fraction = 1 / 7
cutoff_index = int(len(runtime) * top_fraction)
top_iterations = iterations[cutoff_index:]
top_runtime = runtime[cutoff_index:]

# Perform a linear fit in the log-log space on the top 6/7 of the data
log_top_iterations = np.log10(top_iterations)
log_top_runtime = np.log10(top_runtime)

slope, intercept, _, _, _ = linregress(log_top_runtime, log_top_iterations)

# Calculate the fitted line for the entire range
fitted_iterations = 10**(slope * np.log10(runtime) + intercept)

# Calculate residuals for the top 6/7 and standard deviation of residuals
log_fitted_top_iterations = slope * log_top_runtime + intercept
residuals_top = log_top_iterations - log_fitted_top_iterations
std_dev_residuals = np.std(residuals_top)

# Identify points that deviate more than 5 sigma in the full dataset
log_fitted_iterations = slope * np.log10(runtime) + intercept
log_actual_iterations = np.log10(iterations)
residuals_all = log_actual_iterations - log_fitted_iterations
outliers = np.abs(residuals_all) > 5 * std_dev_residuals

# Determine the cutoff runtime below which there are outliers
cutoff_runtime = runtime[np.argmax(outliers)] if np.any(outliers) else None

# Plot data, linear fit, 5 sigma bounds, and mark the cutoff point
plt.figure(figsize=(6, 6))
plt.plot(runtime, iterations, marker='.', linestyle='-', color='b', label='Data')
plt.plot(runtime, fitted_iterations, color='r', linestyle='--', label='Linear Fit (Top 6/7)')
plt.scatter(runtime[outliers], iterations[outliers], color='orange', label='5σ Outliers')

# Add 5 sigma bounds
upper_bound = 10**(log_fitted_iterations + 5 * std_dev_residuals)
lower_bound = 10**(log_fitted_iterations - 5 * std_dev_residuals)
plt.plot(runtime, upper_bound, color='g', linestyle=':', label='+5σ Bound')
plt.plot(runtime, lower_bound, color='g', linestyle=':', label='-5σ Bound')

largest_outlier = 0
for i in np.where(outliers)[0]:
    if runtime[i] > largest_outlier:
        largest_outlier = runtime[i]


# Mark the cutoff point if it exists
if cutoff_runtime:
    plt.axvline(x=largest_outlier, color='purple', linestyle='--', label=f'Largest Outlier ≈ {largest_outlier:.5f}s')

plt.xlabel('Runtime [s]')
plt.ylabel('Iterations')
plt.title('Iteration Runtime Proportionality with 5σ Bounds')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('plots/proport/A30_with_fit_and_sigma.png', format='png')
plt.savefig('plots/proport/A30_with_fit_and_sigma.pdf', format='pdf')


# Print the number of outliers and the actual data points
print(f"Number of points outside 5σ range: {np.sum(outliers)}")
print("Outliers (Runtime [s], Iterations):")
for i in np.where(outliers)[0]:
    print(f"({runtime[i]:.5f}, {iterations[i]})")


