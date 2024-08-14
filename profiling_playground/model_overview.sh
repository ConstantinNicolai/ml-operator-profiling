#!/bin/bash

operator  = "conv" 

python3 print.py | grep $operator >> temp_print_$operator

python3 tree.py | grep -o "$operator.*" >> temp_tree_$operator

