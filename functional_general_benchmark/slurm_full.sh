#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=full_model_RTX2080TI
#SBATCH --output=dataset_history_RTX2080TI/full_model_measurements_RTX2080TI.txt
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


# kill_background_jobs() {
#     for pid in $@; do
#         kill $pid
#     done
# }


python3 full_model_measurement.py