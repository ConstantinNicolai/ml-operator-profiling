#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <your_variable>"
    exit 1
fi

# Assign the first command-line argument to a variable
TEST="$1"

python3 sum_up_from_dataset_train.py --path=$TEST > temp_ops_$TEST


python3 fullmodel_output.py --path=$TEST > temp_fm_$TEST


python3 plot_comp_alphabetical.py temp_fm_$TEST temp_ops_$TEST $TEST

