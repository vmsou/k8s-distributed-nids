#!/bin/bash

FOLDS=5
METRIC="f1"
dataset_names=("NB15" "ToN" "IDS" "BoT")
datasets="datasets/NF-UNSW-NB15-v2.parquet datasets/NF-CSE-CIC-IDS2018-v2.parquet datasets/NF-ToN-IoT-v2.parquet datasets/NF-BoT-IoT-v2.parquet"

model_order=("DT" "GBT" "LR" "MLP" "RF")

base_command="spark-submit --executor-memory 9g apps/evaluate-model.py -d $datasets -m %s --log %s"

for dataset_name in "${dataset_names[@]}"; do
  models=""
  for model in "${model_order[@]}"; do
    models+="models/${model}/${model}_${FOLDS}F_${METRIC}_${dataset_name}.model "
  done

  log_file="metrics/${dataset_name}_metrics.csv"
  
  eval_command=$(printf "$base_command" "$models" "$log_file")
  eval "$eval_command"
done
