#!/bin/bash

FOLDS=5
METRIC="areaUnderPR"
EXECUTOR_MEMORY="7g"
base_command="spark-submit --executor-memory $EXECUTOR_MEMORY apps/train-model.py cross-validator --folds $FOLDS --metric $METRIC -s '%s' -d '%s' -o '%s' --log metrics/training_data.csv"

# Model order list
model_order=("DT" "GBT" "LR" "MLP" "RF")

# Model setup dictionary
declare -A model_setups=(
  ["DT"]="setups/DT_NETV2.setup"
  ["GBT"]="setups/GBT_NETV2.setup"
  ["LR"]="setups/LR_NETV2.setup"
  ["MLP"]="setups/MLP_NETV2.setup"
  ["RF"]="setups/RF_NETV2.setup"
)

# Dataset names and paths
dataset_names=("NB15" "ToN" "IDS" "BoT")

declare -A datasets=(
  ["NB15"]="datasets/NF-UNSW-NB15-v2.parquet"
  ["ToN"]="datasets/NF-ToN-IoT-v2.parquet"
  ["IDS"]="datasets/NF-CSE-CIC-IDS2018-v2.parquet"
  ["BoT"]="datasets/NF-BoT-IoT-v2.parquet"
)

for dataset_name in "${dataset_names[@]}"; do
  dataset=${datasets[$dataset_name]}
  
  for model in "${model_order[@]}"; do
    setup=${model_setups[$model]}
    command=$(printf "$base_command" "$setup" "$dataset" "models/${model}/${model}_${FOLDS}F_${METRIC}_${dataset_name}.model")
    eval "$command"
  done
done
