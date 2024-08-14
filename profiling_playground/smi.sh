#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=smi_meas
#SBATCH --output=resnet18_print.out
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass




srun python3 resnet.py