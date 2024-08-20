#!/bin/bash

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass

operator="conv"

python3 model_analysis_print.py | grep $operator > temp_print_$operator

python3 model_analysis_bigtree.py | grep -o "$operator.*" > temp_tree_$operator

sleep 20

python3 filter_print_results.py temp_print_$operator

python3 filter_bigtree_results.py temp_tree_$operator

python3 merge_dicts.py temp_print_$operator.json temp_tree_$operator.json resnet50_$operator