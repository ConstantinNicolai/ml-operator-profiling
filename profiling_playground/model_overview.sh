#!/bin/bash

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate constabass

operator="conv" 

python3 print.py | grep $operator >> temp_print_$operator

python3 tree.py | grep -o "$operator.*" >> temp_tree_$operator

