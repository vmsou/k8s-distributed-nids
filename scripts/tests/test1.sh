#!/bin/bash

FOLDS=2
dataset_names=("NB15" "ToN" "IDS" "BoT")
datasets="datasets/NF-UNSW-NB15-v2.parquet datasets/NF-CSE-CIC-IDS2018-v2.parquet datasets/NF-ToN-IoT-v2.parquet datasets/NF-BoT-IoT-v2.parquet"

base_command="spark-submit --conf spark.storage.memoryFraction=0.3 --executor-memory 4g apps/evaluate-model.py -d $datasets -m %s --log %s"

for dataset_name in "${dataset_names[@]}"; do
  # Dynamically build model paths
  models="models/DT_${FOLDS}F_PRAUC_${dataset_name}.model models/GBT_${FOLDS}F_PRAUC_${dataset_name}.model models/RF_${FOLDS}F_PRAUC_${dataset_name}.model models/LR_${FOLDS}F_PRAUC_${dataset_name}.model models/MLP_${FOLDS}F_PRAUC_${dataset_name}.model"
  log_file="metrics/${dataset_name}_metrics.csv"
  eval_command=$(printf "$base_command" "$models" "$log_file")
  eval "$eval_command"
done
