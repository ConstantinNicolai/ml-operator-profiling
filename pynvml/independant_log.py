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
    logger = GPUStatLogger(device_index=0, log_interval=0.5, log_file_path="gpu_usage_log.txt")
    
    # Start logging
    logger.start()
    
    # Run the PyTorch workload
    try:
        run_pytorch_workload()
    finally:
        # Ensure that logging stops even if an error occurs
        logger.stop()
