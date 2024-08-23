# Distributed NIDS

## Quick Start
This project involves Distributed Network Intrusion Detection System (NIDS) with Big Data and KDD. The main goal is to create an efficient system that is scalable and capable of identifying intrusions in large-scale networks using deep learning and distributed processing.

It is recommended to use a Linux environment for better compatibility and performance.

### Components
- Apache Spark: Used for parallel processing and real-time data analysis and MLlib for Machine Learning
- HDFS (Hadoop Distributed File System): For distributed storage
- Apache Kafka: Responsible for continuous ingestion of network data

### Software
- [Kubernetes](https://kubernetes.io/)
- [Helm](https://helm.sh/)
- [Minikube (Optional)](https://minikube.sigs.k8s.io/)

### Setting the Environment (with Minikube)
```bash
minikube start --nodes 1 --cpus 7 --memory 7g --disk-size 30g --driver hyperv --profile dnids
```

### Deploy (with Minikube)
To deploy the NIDS cluster, run:
```bash
helm install dnids charts/dnids
```

To exclude components from being deployed you can use --set <component>.enabled=false
```bash
helm install dnids charts/dnids --set kafka.enabled=false
helm upgrade dnids charts/dnids --set kafka.enabled=true
```

### Port-Forward
```bash
kubectl port-forward svc/hdfs-namenode-svc 9870:9870
kubectl port-forward svc/spark-master-svc 8080:8080
```

### Interface URLs
- HDFS UI: http://localhost:9870
- Spark Master UI: http://localhost:8080

## Notebooks - Machine Learning
This project utilizes Spark MLlib. To prepare your own models, you can use the following links:
- [Setup Model](https://colab.research.google.com/drive/10v5uXBmioFk7bZeAtYbnHJ6-CS7OSq6U?usp=sharing)
- [Train Model](https://colab.research.google.com/drive/1V2kn61Jl1Hhnuv0KJpcvR3S6pqYt_2uE?usp=sharing)
- [Evaluate Model](https://colab.research.google.com/drive/1hrTI9o2uxjBrOD2hzKI_gn3sC5Rap-1Z?usp=sharing)

This "pipeline" enables to save each step of the model for reproducibility. You can also train and evaluate a model inside of the distributed
infrastructure with Spark and HDFS with the apps `train-model.py` and `evaluate-model.py`, but setups files are necessary for these, which can be
made using the link above.


### Configuring HDFS
```bash
kubectl exec -ti hdfs-namenode-0 -- bash
hdfs dfs -mkdir -p /user/ /tmp/ /user/spark/ /user/spark/models /user/spark/datasets/ /user/spark/schemas /user/spark/setups
hdfs dfs -chmod 755 /user/
hdfs dfs -chmod -R 777 /tmp/
hdfs dfs -chown -R spark:spark /user/spark/
hdfs dfs -chmod -R 775 /user/spark/
```

### Configuring HDFS - Adding files (if needed)
- Schemas: JSON files of the DataFrame Schema
- Datasets: CSV or Parquet files
- Setups: Folders generated from saving a model before training (see setup model notebook)
- Models: Spark MLlib Models

```bash
hdfs dfs -put <schema> /user/spark/schemas/
hdfs dfs -put <dataset> /user/spark/datasets/
hdfs dfs -put <setup> /user/spark/setups/
hdfs dfs -put <model> /user/spark/models/
```

### Configuring Kafka
```bash
kubectl exec -ti kafka-broker-0 -- bash
kafka-topics.sh --create --topic NetV2 --bootstrap-server kafka-broker-svc:9092
kafka-topics.sh --list --bootstrap-server kafka-broker-svc:9092
kafka-console-producer.sh --broker-list kafka-broker-svc:9092 --topic NetV2
>csv,separated,data
```

### Submitting Application to Spark
```bash
# Training a model
kubectl exec -ti spark-worker-0 -- spark-submit apps/train-model.py cross-validator --folds 2 -p 5 -s "setups/<setup_file>" -d "datasets/<dataset_file>" -o "models/<model_file>"

# Predictions
kubectl exec -ti spark-worker-0 -- spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 apps/kafka-predictions.py -b kafka-broker-0.kafka-headless.default.svc.cluster.local:9092 -t NetV2 --model "models/<model_file>" --schema schemas/<schema_file>
```
