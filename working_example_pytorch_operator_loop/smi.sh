#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=smi_meas
#SBATCH --output=rolling_output_nojobnumber.out
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


kill_background_jobs() {
    for pid in $@; do
        kill $pid
    done
}


# Main script

# Run the benchmark
srun python3 attempt_4.py --in_channels 32 --out_channels 64 --kernel_size 5 --stride 2 --padding 0 --batch_size 16 --ifmap_size 28 >> logs/training_output_${SLURM_JOB_ID}.log

srun python3 attempt_4.py --ifmap_size 32 >> logs/training_output_${SLURM_JOB_ID}.log
