#!/bin/bash

#SBATCH --partition=all
#SBATCH --nodes=1 
#SBATCH --job-name=operator_profiling
#SBATCH --output=A30_no_tc_full_model
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass



python3 full_model_meas.py --gpu=A30_no_tc