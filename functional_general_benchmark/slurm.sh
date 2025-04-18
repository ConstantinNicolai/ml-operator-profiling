#!/bin/bash

#SBATCH --partition=rivulet
#SBATCH --nodes=1 
#SBATCH --job-name=A30vm1440
#SBATCH --output=A30vm1440.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:1
#SBATCH --time=168:00:00




# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass



sudo /opt/csg/scripts/nvidia-set-clocks.sh -d 0 -c 1440 &
CLOCK_PID=$!;
printf "setting clock in the backround at PID %s\n" $CLOCK_PID
sleep 5;
echo "Query clock to confirm"
nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv;

python3 full_model_meas_infvalidation.py  --gpu=A30_1440 --rundur=14 --runnr=20
# python3 full_model_meas_train.py --gpu=A30_210 --rundur=12 --runnr=60
# python3 general_pipeline_block2_backprop.py --gpu=A30_210 --rundur=10 --runnr=25


echo "Kill clock script and query clock again"
kill $CLOCK_PID;


# python3 general_pipeline_block2_backprop.py --gpu=RTX2080TI_210_405 --rundur=10 --runnr=22

# python3 full_model_meas_train.py --gpu=A30 --rundur=12 --runnr=4

# python3 min_ex_copy.py




