#!/bin/bash

#SBATCH --partition=all
#SBATCH --job-name=auto_benchmark
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

# Define the JSON file
json_file="../operator_search/resnet34_conv_merged.json"


# Read and process the JSON file using jq
in_channels_list=($(jq -r '.[] | .in_channels' $json_file))
out_channels_list=($(jq -r '.[] | .out_channels' $json_file))
kernel_size_list=($(jq -r '.[] | .kernel_size' $json_file))
stride_list=($(jq -r '.[] | .stride' $json_file))
padding_list=($(jq -r '.[] | .padding' $json_file))
batch_size_list=($(jq -r '.[] | .count | tonumber | if . == 0 then 32 else 32 end' $json_file)) # Default batch size is 32
ifmap_size_list=($(jq -r '.[] | .input_size' $json_file))

# Print the lists (for verification)
echo "in_channels_list=(${in_channels_list[@]})"
echo "out_channels_list=(${out_channels_list[@]})"
echo "kernel_size_list=(${kernel_size_list[@]})"
echo "stride_list=(${stride_list[@]})"
echo "padding_list=(${padding_list[@]})"
echo "batch_size_list=(${batch_size_list[@]})"
echo "ifmap_size_list=(${ifmap_size_list[@]})"


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
        --iterations 90000 \
        >> logs/benchmark_durations_${SLURM_JOB_ID}.log
done
