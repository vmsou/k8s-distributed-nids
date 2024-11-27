#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model1> [model2] [model3] ..."
  exit 1
fi

models=$(printf "%s " "$@" | sed 's/ $//')
datasets="datasets/NF-UNSW-NB15-v2.parquet datasets/NF-ToN-IoT-v2.parquet datasets/NF-CSE-CIC-IDS2018-v2.parquet datasets/NF-BoT-IoT-v2.parquet"
base_command="spark-submit --conf spark.storage.memoryFraction=0.3 --executor-memory 4g apps/evaluate-model.py -d $datasets -m $models --ensemble %s --log metrics/%s_ensemble.csv"
ensemble_types=("majority" "attack" "normal")

for ensemble_type in "${ensemble_types[@]}"; do  
    command=$(printf "$base_command" "$ensemble_type" "$ensemble_type")
    eval "$command"
done
