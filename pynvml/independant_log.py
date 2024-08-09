import threading
import time
import torch
import torch.nn as nn
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
        self.device_index = device_index
        self.log_interval = log_interval
        self.log_file_path = log_file_path
        self._stop_event = threading.Event()
        self._logger_thread = threading.Thread(target=self._log_stats)
    
    def start(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(self.device_index)
        with open(self.log_file_path, "w") as log_file:
            log_file.write("Timestamp,GPU Usage (%),Memory Usage (MB),Power Usage (W)\n")
        self._logger_thread.start()
        print("GPU logging started.")
    
    def stop(self):
        self._stop_event.set()
        self._logger_thread.join()
        nvmlShutdown()
        print("GPU logging stopped.")
    
    def _log_stats(self):
        while not self._stop_event.is_set():
            timestamp = time.time()
            memory_info = nvmlDeviceGetMemoryInfo(self.handle)
            utilization = nvmlDeviceGetUtilizationRates(self.handle)
            
            power_usage = None
            for _ in range(3):
                try:
                    power_usage = nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    break
                except Exception:
                    time.sleep(0.1)
                    continue
            
            if power_usage is not None:
                log_entry = f"{timestamp},{utilization.gpu},{memory_info.used / 1024**2},{power_usage}\n"
            else:
                log_entry = f"{timestamp},{utilization.gpu},{memory_info.used / 1024**2},N/A\n"
            
            with open(self.log_file_path, "a") as log_file:
                log_file.write(log_entry)

            time.sleep(self.log_interval)

# Adjusted Intensive GPU Workload
def run_intensive_gpu_workload():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Reduce tensor sizes to avoid memory errors
    large_tensor1 = torch.randn((512, 512, 512), device=device)  # Smaller tensor
    large_tensor2 = torch.randn((512, 512, 512), device=device)  # Smaller tensor
    
    # Increase the number of iterations to maintain intensity
    iterations = 5000
    start_time = time.time()
    
    for i in range(iterations):
        # Perform matrix multiplication
        result = torch.matmul(large_tensor1, large_tensor2)
        del result
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for {iterations} iterations: {total_time:.4f} seconds")

if __name__ == "__main__":
    # Initialize the GPUStatLogger
    logger = GPUStatLogger(device_index=0, log_interval=0.5, log_file_path="gpu_usage_log.txt")
    
    # Start logging
    logger.start()
    
    # Run the intensive GPU workload
    try:
        run_intensive_gpu_workload()
    finally:
        # Ensure that logging stops even if an error occurs
        logger.stop()
