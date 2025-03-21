#!/bin/bash

#SBATCH --partition=brook
#SBATCH --nodes=1 
#SBATCH --job-name=555op405
#SBATCH --output=RTX2080TI_555op405.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1
#SBATCH --time=96:00:00




# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass




sudo /opt/csg/scripts/nvidia-set-clocks.sh -d 0 -c 555 -m 405 &
CLOCK_PID=$!;
printf "setting clock in the backround at PID %s\n" $CLOCK_PID
sleep 5;
echo "Query clock to confirm"
nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv;

python3 general_pipeline_block2_backprop.py --gpu=RTX2080TI_555_405 --rundur=10 --runnr=22


echo "Kill clock script and query clock again"
kill $CLOCK_PID;



# python3 full_model_meas_train.py --gpu=A30 --rundur=12 --runnr=4

# python3 min_ex_copy.py




# python3 full_model_meas_trainvalidation.py  -gpu=A30 --rundur=14 --runnr=5