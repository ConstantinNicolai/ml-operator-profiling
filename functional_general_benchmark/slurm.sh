#!/bin/bash

#SBATCH --partition=rivulet
#SBATCH --nodes=1 
#SBATCH --job-name=operations_profiling
#SBATCH --output=clocks.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:1
#SBATCH --time=96:00:00
#SBATC --mem=32G



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass



# python3 full_model_meas_train.py --gpu=A30 --rundur=12 --runnr=4

# python3 min_ex_copy.py

# python3 general_pipeline_block2_backprop.py --gpu=RTX2080TI --rundur=10 --runnr=5


# python3 full_model_meas_trainvalidation.py  --gpu=A30 --rundur=20 --runnr=5

# python3 general_pipeline_block1_predictions.py

python3 clockscript.py
