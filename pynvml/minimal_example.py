import torch
import torch.nn as nn
import torch.optim as optim
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetUtilizationRates, nvmlShutdown

# Initialize NVML
nvmlInit()
device_index = 0  # You can change this to the appropriate GPU index
handle = nvmlDeviceGetHandleByIndex(device_index)

# Create a log file
log_file = open("gpu_usage_log.txt", "w")

# Write the header of the log file
log_file.write("Time, GPU Usage (%), Memory Usage (MB), Power Usage (W)\n")

# Simple PyTorch model and data for demonstration
model = nn.Linear(1000, 1000).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dummy input and target tensors
inputs = torch.randn(64, 1000).cuda()
targets = torch.randn(64, 1000).cuda()

# Number of iterations to simulate a long-running process
num_iterations = 1000

for iteration in range(num_iterations):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log GPU usage, memory usage, and power usage
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    utilization = nvmlDeviceGetUtilizationRates(handle)
    power_usage = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from milliwatts to watts
    
    log_file.write(f"{time.time()}, {utilization.gpu}, {memory_info.used / 1024**2}, {power_usage}\n")
    
    # Sleep to simulate time between iterations (optional, to extend the runtime)
    time.sleep(0.1)

# Close the log file
log_file.close()

# Shutdown NVML
nvmlShutdown()
