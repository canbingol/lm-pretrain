#!/bin/bash

config=$1
param_count=$#

# ---- Check number of parameters ----
if [ "$param_count" -ne 1 ]; then
    echo "expected number of params 1 but given $param_count"
    echo "user params-> "$@""
    exit 1
fi

WORLD_SIZE=$(grep -E "^[[:space:]]*world_size:" "$config" | awk '{print $2}')
echo "World size: $WORLD_SIZE"
torchrun --standalone --nproc_per_node="$WORLD_SIZE" main.py --config "$config"
