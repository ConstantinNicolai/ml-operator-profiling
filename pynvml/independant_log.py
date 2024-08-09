import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetUtilizationRates,
)

class GPUStatLogger:
    def __init__(self, device_index=0, log_interval=1.0, log_file_path="gpu_usage_log.txt"):
        """
        Initializes the GPUStatLogger.

        Args:
            device_index (int): Index of the GPU to monitor.
            log_interval (float): Time interval (in seconds) between log entries.
            log_file_path (str): Path to the log file.
        """
        self.device_index = device_index
        self.log_interval = log_interval
        self.log_file_path = log_file_path
        self._stop_event = threading.Event()
        self._logger_thread = threading.Thread(target=self._log_stats)
    
    def start(self):
        """
        Starts the logging thread.
        """
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(self.device_index)
        with open(self.log_file_path, "w") as log_file:
            log_file.write("Timestamp,GPU Usage (%),Memory Usage (MB),Power Usage (W)\n")
        self._logger_thread.start()
        print("GPU logging started.")
    
    def stop(self):
        """
        Signals the logging thread to stop and waits for it to finish.
        """
        self._stop_event.set()
        self._logger_thread.join()
        nvmlShutdown()
        print("GPU logging stopped.")
    
    def _log_stats(self):
        """
        The target function for the logging thread. Logs GPU stats at regular intervals.
        """
        while not self._stop_event.is_set():
            timestamp = time.time()
            memory_info = nvmlDeviceGetMemoryInfo(self.handle)
            utilization = nvmlDeviceGetUtilizationRates(self.handle)
            power_usage = nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert from milliwatts to watts

            log_entry = f"{timestamp},{utilization.gpu},{memory_info.used / 1024**2},{power_usage}\n"

            with open(self.log_file_path, "a") as log_file:
                log_file.write(log_entry)

            time.sleep(self.log_interval)

# Example PyTorch Workload
def run_pytorch_workload():
    """
    A dummy PyTorch workload to simulate GPU activity.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(1000, 1000).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dummy data
    inputs = torch.randn(64, 1000).to(device)
    targets = torch.randn(64, 1000).to(device)

    num_iterations = 1000
    for _ in range(num_iterations):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # Introduce a small sleep to simulate processing time
        time.sleep(0.01)

if __name__ == "__main__":
    # Initialize the GPUStatLogger
    logger = GPUStatLogger(device_index=0, log_interval=0.001, log_file_path="gpu_usage_log.txt")
    
    # Start logging
    logger.start()
    
    # Run the PyTorch workload
    try:
        #run_pytorch_workload()
        # Configuration for the convolutional layer
        in_channels = 64
        out_channels = 128
        kernel_size = 3
        stride = 1
        padding = 1

        # Assume the input tensor size after initial downsampling is (batch_size, 64, 56, 56)
        batch_size = 32
        input_size = (batch_size, in_channels, 56, 56)

        # Create a large array of random convolutional layers stored in VRAM
        num_layers = 30000  # Large number of layers to simulate a large model
        conv_layers = []
        for _ in range(num_layers):
            layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding).cuda()
            conv_layers.append(layer)

        # Create a large array of random input data on the GPU
        data_size = (1000,) + input_size[1:]  # Large dataset to simulate caching scenario
        input_data = torch.randn(data_size).cuda()

        # Number of iterations to run
        iterations = 15000000

        # Start the timer
        start_time = time.time()

        # Run the convolution operation in a loop, accessing layers linearly
        for i in range(iterations):
            # Linearly access the convolutional layer from the pre-created list
            conv_layer = conv_layers[i % num_layers]
            
            # Index into the data array, using modulo to loop over if necessary
            index = i % data_size[0]
            x = input_data[index:index+1]
            
            # Apply the convolution operation
            output = conv_layer(x)

        # Stop the timer
        end_time = time.time()

        # Calculate the time taken
        total_time = end_time - start_time
        print(f"Total time for {iterations} iterations: {total_time:.4f} seconds")

    finally:
        # Ensure that logging stops even if an error occurs
        logger.stop()
