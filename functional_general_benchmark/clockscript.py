import subprocess
import signal
import os
import time 

time.sleep(4)


# processes = []



# command = (
#     f"sudo "
#     f"/opt/csg/scripts/nvidia-set-clocks.sh -c 810 -d 0 "
#     f"> clocks.log"
# )
 
# process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
# processes.append(process)
# print("take some time")
# print("tooday")
# time.sleep(1)
# os.killpg(os.getpgid(processes[-1].pid), signal.SIGTERM)  # Use processes[-1] to get the last process

# import subprocess
# import signal
# import os
# import time 

# processes = []

# command = [
#     "sudo",
#     "/opt/csg/scripts/nvidia-set-clocks.sh",
#     "-c", "810",
#     "-d", "0"
# ]

# with open("clocks.log", "w") as logfile:
#     process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.PIPE, preexec_fn=os.setsid)
#     processes.append(process)

# command0 = ["nvidia-smi",  "--query-gpu=clocks.mem,clocks.gr", "--format=csv"]


# with open("clockspeed.txt", "w") as logfile:
#     process = subfprocess.Popen(command0, stdout=logfile, stderr=subprocess.PIPE, preexec_fn=os.setsid)
#     processes.append(process)

# print("take some time")
# time.sleep(5)  # Increase sleep time if needed

# command0 = ["nvidia-smi",  "--query-gpu=clocks.mem,clocks.gr", "--format=csv"]


# with open("clockspeed.txt", "w") as logfile:
#     process = subprocess.Popen(command0, stdout=logfile, stderr=subprocess.PIPE, preexec_fn=os.setsid)
#     processes.append(process)

# time.sleep(2)

# # Gracefully terminate the process group
# os.killpg(os.getpgid(processes[-1].pid), signal.SIGTERM)

# # Gracefully terminate the process group
# os.killpg(os.getpgid(processes[-1].pid), signal.SIGTERM)


