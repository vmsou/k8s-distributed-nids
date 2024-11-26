#!/bin/bash

cores=(10 8 6 4 2)
FOLDS=2
base_command="spark-submit --conf spark.storage.memoryFraction=0.3 --executor-memory 4g --conf spark.executor.cores=2 --conf spark.cores.max=%d apps/train-model.py cross-validator --folds $FOLDS --metric areaUnderPR -s '%s' -d '%s' -o '%s' --log metrics/training_data.csv"

# Model order list
model_order=("DTC" "GBT" "LR" "MLPC" "RF")

# Model setup dictionary
declare -A model_setups=(
  ["DTC"]="setups/SETUP_NETV2_DTC_PCA10_CV_PR_AUC"
  ["GBT"]="setups/SETUP_NETV2_GBT_PCA10_CV_PR_AUC"
  ["LR"]="setups/SETUP_NETV2_LR_PCA10_CV_PR_AUC"
  ["MLPC"]="setups/SETUP_NETV2_MLPC_PCA10_LAYER_CV_PR_AUC"
  ["RF"]="setups/SETUP_NETV2_RF_PCA10_CV_PR_AUC"
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
  
  for core_count in "${cores[@]}"; do
    for model in "${model_order[@]}"; do
      setup=${model_setups[$model]}
      command=$(printf "$base_command" "$core_count" "$setup" "$dataset" "models/${model}_${FOLDS}F_PRAUC_${dataset_name}.model")
      eval "$command"
    done
  done
done
