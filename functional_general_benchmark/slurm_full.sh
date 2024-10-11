#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=full_model_A30
#SBATCH --output=dataset_history_A30_no_tc/full_model_measurements_A30_no_tc.txt
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


# kill_background_jobs() {
#     for pid in $@; do
#         kill $pid
#     done
# }


python3 full_model_measurement.py