#!/bin/bash
#SBATCH --job-name=important
#SBATCH --output=output_resub.log
#SBATCH --time=48:05:00  # Maximum run time of 5 minutes
#SBATCH --ntasks=1        # Number of tasks
#SBATCH --cpus-per-task=1 # Number of CPU cores per task
#SBATCH --mem=1G          # Memory allocation
#SBATCH --partition=all

# User variables
USERNAME="kx225"  # Replace <username> with the actual username
JOB_ID="$SLURM_JOB_ID"  # Get the current job ID

# Infinite loop
while true; do
    # Check if the output of the command is empty
    if [ -z "$(squeue -u "$USERNAME" | grep -v "$JOB_ID")" ]; then
        # If it's empty, submit the sbatch job
        sbatch slurm_RTX2080TI.sh
        echo "Submitted test.sh at $(date)"
    else
        echo "Jobs are still running for user $USERNAME. Waiting..."
    fi

    # Sleep for 3 hours (10800 seconds)
    sleep 10800
done
