#!/bin/bash

# Set the number of folds
FOLDS=3

# Define dataset names and evaluation datasets
dataset_names=("NB15" "ToN" "IDS" "BoT")
all_datasets="datasets/NF-UNSW-NB15-v2.parquet datasets/NF-CSE-CIC-IDS2018-v2.parquet datasets/NF-ToN-IoT-v2.parquet datasets/NF-BoT-IoT-v2.parquet"

# Base evaluation command
base_command="spark-submit apps/evaluate-model.py -d $all_datasets -m %s --log %s"

for dataset_name in "${dataset_names[@]}"; do
  # Dynamically build model paths
  models="models/DTC_${FOLDS}F_PRAUC_${dataset_name}.model models/GBT_${FOLDS}F_PRAUC_${dataset_name}.model models/RF_${FOLDS}F_PRAUC_${dataset_name}.model models/LR_${FOLDS}F_PRAUC_${dataset_name}.model models/MLPC_${FOLDS}F_PRAUC_${dataset_name}.model"
  log_file="metrics/${dataset_name}_metrics.csv"
  eval_command=$(printf "$base_command" "$models" "$log_file")
  eval "$eval_command"
done
