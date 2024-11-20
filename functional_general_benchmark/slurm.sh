#!/bin/bash

#SBATCH --nodelist=csg-brook02
#SBATCH --nodes=1 
#SBATCH --job-name=operations_profiling
#SBATCH --output=fullmodel_RTX2080TI
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass



python3 full_model_meas.py --gpu=RTX2080TI --rundur=12 --runnr=3