#!/bin/bash

#SBATCH --nodelist=csg-brook02
#SBATCH --nodes=1 
#SBATCH --job-name=operations_profiling
#SBATCH --output=operations_RTX2080TI_train
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1
#SBATCH --time=48:00:00



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass



python3 general_pipeline_block2_backprop.py --gpu=RTX2080TI --rundur=8 --runnr=4
