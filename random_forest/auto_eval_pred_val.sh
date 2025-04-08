#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <your_variable>"
    exit 1
fi

# Assign the first command-line argument to a variable
TEST="$1"

echo $TEST

ls ../functional_general_benchmark/datasets_fullmodel_train_validation/dataset_history_A30_$TEST

ls ../functional_general_benchmark/datasets_fullmodel_train_validation/dataset_history_A30_210

python3 ../functional_general_benchmark/fullmodel_output.py --path=A30_$TEST > ../functional_general_benchmark/datasets_fullmodel_train_validation/dataset_history_A30_$TEST/fullmodel.txt


python3 model_prediction_forall.py --clock=$TEST --mode=train > ../functional_general_benchmark/datasets_fullmodel_train_validation/dataset_history_A30_$TEST/prediction.txt


python3 ../functional_general_benchmark/plot_comparison.py --clock=$TEST 


# python3 sum_up_from_dataset_train.py --path=$TEST > temp_ops_$TEST


# python3 fullmodel_output.py --path=$TEST > temp_fm_$TEST


# python3 plot_comp_alphabetical.py temp_fm_$TEST temp_ops_$TEST $TEST

# python3 plot_time_alphabetical.py temp_fm_$TEST temp_ops_$TEST $TEST