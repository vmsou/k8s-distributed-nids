models="models/DTC_3F_PRAUC_NB15.model models/GBT_3F_PRAUC_ToN.model models/RF_3F_PRAUC_IDS.model"
datasets="datasets/NF-UNSW-NB15-v2.parquet datasets/NF-ToN-IoT-v2.parquet datasets/NF-CSE-CIC-IDS2018-v2.parquet datasets/NF-BoT-IoT-v2.parquet"

ensemble_types=("majority" "attack" "normal")
for ensemble_type in "${ensemble_types[@]}"; do  
  eval "spark-submit --executor-memory 4g apps/evaluate-model.py -d $datasets -m $models --ensemble $ensemble_type --log metrics/${ensemble_type}_ensemble.csv"
done
