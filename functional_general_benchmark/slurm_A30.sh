#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=operator_profiling
#SBATCH --output=operator_profiling_A30.out
#SBATCH --error=operator_profiling_A30.er
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


# kill_background_jobs() {
#     for pid in $@; do
#         kill $pid
#     done
# }

python3 general_pipeline_block2_A30.py
