#!/bin/bash

#SBATCH --nodelist=csg-rivulet02
#SBATCH --nodes=1 
#SBATCH --job-name=operations_profiling
#SBATCH --output=operations_A30
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass



python3 general_pipeline_block2.py --gpu=A30 --rundur=4 --runnr=4