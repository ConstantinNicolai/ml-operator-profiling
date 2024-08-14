#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=smi_meas
#SBATCH --output=rolling_output_nojobnumber.out
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX2080TI:1



# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass


kill_background_jobs() {
    for pid in $@; do
        kill $pid
    done
}


# Main script


# Define lists for each parameter
in_channels_list=(3 64 64 64 256 256 128 128 64 64)
out_channels_list=(64 64 64 256 64 128 128 512 128 128)
kernel_size_list=(7 1 3 1 1 1 3 1 3 3)
stride_list=(2 1 1 1 1 1 2 1 1 1 )
padding_list=(3 0 1 0 0 0 1 0 1 1 )
batch_size_list=(32 32 32 32 32 32 32 32 32 32)
ifmap_size_list=(224 56 56 56 56 56 56 56 56 56)



num_combinations=${#in_channels_list[@]}
# Run the benchmark for each indexed combination of parameters
for ((i=0; i<$num_combinations; i++)); do
    in_channels=${in_channels_list[$i]}
    out_channels=${out_channels_list[$i]}
    kernel_size=${kernel_size_list[$i]}
    stride=${stride_list[$i]}
    padding=${padding_list[$i]}
    batch_size=${batch_size_list[$i]}
    ifmap_size=${ifmap_size_list[$i]}

    echo "Running: in_channels=$in_channels, out_channels=$out_channels, kernel_size=$kernel_size, stride=$stride, padding=$padding, batch_size=$batch_size, ifmap_size=$ifmap_size"
    srun python3 attempt_4.py \
        --in_channels "$in_channels" \
        --out_channels "$out_channels" \
        --kernel_size "$kernel_size" \
        --stride "$stride" \
        --padding "$padding" \
        --batch_size "$batch_size" \
        --ifmap_size "$ifmap_size" \
        >> logs/training_output_${SLURM_JOB_ID}.log
done



# # Run the benchmark
# srun python3 attempt_4.py --in_channels 128 --out_channels 256 --kernel_size 7 --stride 1 --padding 1 --batch_size 64 --ifmap_size 56 >> logs/training_output_${SLURM_JOB_ID}.log

# srun python3 attempt_4.py --ifmap_size 32 >> logs/training_output_${SLURM_JOB_ID}.log
