#!/bin/bash

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass

operator="conv"
model="resnet50

python3 print.py | grep $operator >> temp_print_$operator

python3 tree.py | grep -o "$operator.*" >> temp_tree_$operator

python3 dictex_unique_count.py temp_print_$operator

python3 treestudy1.py temp_tree_$operator

python3 merge_dicts.py temp_print_$operator.json temp_tree_$operator.json resnet18_$operator

