#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=operator_profiling
#SBATCH --output=warmup.out
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1


# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


python3 full_model_again.py
