#!/bin/bash

TRAIN_CONFIG="config_yamls/train_config.yaml"
INFERENCE_CONFIG="config_yamls/inference_config.yaml"

type=$1
param_count=$#

# ---- Check number of parameters ----
if [ "$param_count" -ne 1 ]; then
    echo "expected number of params 1 but given $param_count"
    echo "user params-> "$@""
    exit 1
fi

# ---- Type normalization ----
case "$type" in
    pretrain | pre_train | pre-train)
        type="pre-train"
        ;;
    instruction-tuning | instruction_tuning | it)
        type="instruction-tuning"
        ;;
    inference)
        type="inference"
        ;;
    rlhf)
        type="rlhf"
        ;;
    *)
        echo "Invalid type: $type"
        echo "Expected one of: pretrain, pre_train, pre-train, instruction_tuning, instruction-tuning, it, inference, rlhf"
        echo "Example usage: sh run.sh pre-train"
        exit 1
        ;;
esac

# ---- Running according to selected type ----
if [ "$type" = "inference" ]; then
    echo "[INFO] Inference mode"
    CP=$(grep -E "^[[:space:]]*checkpoint:" "$INFERENCE_CONFIG" | awk '{print $2}')
    python inference.py --config "$INFERENCE_CONFIG"

elif [ "$type" = "pre-train" ] || [ "$type" = "instruction-tuning" ]; then
    echo "[INFO] Training mode ($type)"
    WORLD_SIZE=$(grep -E "^[[:space:]]*world_size:" "$TRAIN_CONFIG" | awk '{print $2}')
    echo "World size: $WORLD_SIZE"
    torchrun --standalone --nproc_per_node="$WORLD_SIZE" main.py --config "$TRAIN_CONFIG" --training-type "$type"

elif [ "$type" = "rlhf" ]; then
    echo "[INFO] Training mode ($type)"
    WORLD_SIZE=$(grep -E "^[[:space:]]*world_size:" "$TRAIN_CONFIG" | awk '{print $2}')
    echo "World size: $WORLD_SIZE"
    torchrun --standalone --nproc_per_node="$WORLD_SIZE" rlhf_trainer.py 

else
    echo "Error: Unknown type '$type'."
    exit 1
fi
