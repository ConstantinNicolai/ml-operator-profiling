#!/bin/bash

#SBATCH --partition=all
#SBATCH --nodelist=csg-brook02
#SBATCH --job-name=operator_profiling
#SBATCH --output=logging_consistency.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass



python3 logging_consistency_test.py