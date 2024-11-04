#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=operator_profiling
#SBATCH --output=proportionality_test_A30.out
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass



python3 time_iter_proportion.py