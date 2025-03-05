import subprocess
import signal
import os
import time 


processes = []



command = (
    f"sudo "
    f"/opt/csg/scripts/nvidia-set-clocks.sh -c 810 -d 0 "
    f"> clocks.log"
)
 
process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
processes.append(process)
print("take some time")
time.sleep(1)
os.killpg(os.getpgid(processes[-1].pid), signal.SIGTERM)  # Use processes[-1] to get the last process