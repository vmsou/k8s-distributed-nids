# Distributed NIDS

## Quick Start
This project involves Distributed Network Intrusion Detection System (NIDS) with Big Data and KDD. The main goal is to create an efficient system that is scalable and capable of identifying intrusions in large-scale networks using deep learning and distributed processing.

It is recommended to use a Linux environment for better compatibility and performance.

### README.md structure
- Badges
- Dependences
- Components
- Hardware
- Software
- Environment
- Notebooks (ML)
- Configs

### Considereds badges
- Available Artifacts(SeloD)
- Functional Artifacts (SeloF)
- Sustainteble Artifacts (SeloS)
- Reproductible Experiments (SeloR)

### Dependences 
- Numpy (Last version)
- Openblas
- Spark 3.5.1
- spark-sql-kafka-0-10_2.12
- kafka-clients-3.5.1
- spark-token-provider-kafka-0-10_2.12
- commons-pool2-2.11.1
- HDFS 3.3.6
- Kafka 3.6.2

### Components
- Apache Spark: Used for parallel processing and real-time data analysis and MLlib for Machine Learning
- HDFS (Hadoop Distributed File System): For distributed storage
- Apache Kafka: Responsible for continuous ingestion of network data

### Hardware
- 3 computadores
- Sistema Debian
- 3 Máquinas virtuais hyper-V
- 1x Spark Master (2 Cores, 3GB)
- 5x Spark Worker (2 Cores, 6GB)
- 1x HDFS Namenode (1 Core, 2GB)
- 1x HDFS Datanode (1 Core, 2GB)
- 1x Kafka Broker (2 Cores, 2GB)
- Totalizando 16 Cores e 40GB de RAM rodando no Kubernetes.

### Software
- [Kubernetes](https://kubernetes.io/)
- [Helm](https://helm.sh/)
- [Minikube (Optional)](https://minikube.sigs.k8s.io/)
- [Spark](https://spark.apache.org/)
- [HDFS](https://hadoop.apache.org/)
- [Kafka](https://kafka.apache.org/)

### Setting the Environment

Using Minikube
```bash
minikube start --nodes 1 --cpus 7 --memory 7g --disk-size 30g --driver hyperv --profile dnids
```

### Deploy
To deploy the NIDS cluster, run:
```bash
helm install dnids charts/dnids
```

To exclude components from being deployed you can use --set <component>.enabled=false
```bash
helm install dnids charts/dnids --set kafka.enabled=false
helm upgrade dnids charts/dnids --set kafka.enabled=true
```

You may need a Network Policy like Calico: <br>
https://kubernetes.io/docs/concepts/cluster-administration/addons/
```bash
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
kubectl -n kube-system rollout restart deployment coredns
```

A custom **values.yaml** file may look like this:
```yaml
hdfs:
  enabled: true

  service:
    type: NodePort

  namenode:
    resources: {"requests": {"cpu": "0.5", "memory": "512Mi"}, "limits": {"cpu": "1.0", "memory": "1024Mi"}}
    persistence:
      size: 2G

  datanode:
    replicas: 1

    resources: {"requests": {"cpu": "0.5", "memory": "512Mi"}, "limits": {"cpu": "1.0", "memory": "1024Mi"}}
    persistence:
      size: 10G

spark:
  enabled: true

  service:
    type: NodePort

  master:
    resources: {"requests": {"cpu": "0.5", "memory": "512Mi"}, "limits": {"cpu": "1.0", "memory": "1024Mi"}}

  worker:
    replicas: 1
    resources: {"requests": {"cpu": "0.5", "memory": "512Mi"}, "limits": {"cpu": "1.0", "memory": "1024Mi"}}

kafka:
  enabled: true

  broker:
    replicas: 1

    resources: {"requests": {"cpu": "0.5", "memory": "512Mi"}, "limits": {"cpu": "1.0", "memory": "1024Mi"}}
    persistence:
      size: 2G
```
Then, apply custom values file
```bash
helm install dnids charts/dnids -f values.yaml
helm upgrade dnids charts/dnids -f values.yaml
```

### Scaling
Update values.yaml or using following command:
```
helm upgrade dnids charts/dnids --reuse-values --set spark.worker.replicas=3 --set hdfs.datanode.replicas=3 -- set kafka.broker.replicas=3
```

### Port-Forward
If NodePort or LoadBalancer is not set, use:
```bash
kubectl port-forward svc/hdfs-namenode-svc 9870:9870
kubectl port-forward svc/spark-master-svc 8080:8080
```
else, you can set it using (or updating values.yaml):
```bash
helm install dnids charts/dnids --set spark.service.type=NodePort --set hdfs.service.type=LoadBalancer
```

### Interface Ports
- HDFS UI: http://\<IP\>:9870
- Spark Master UI: http://\<IP\>:8080

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
kafka-topics.sh --create --topic NetV2 --bootstrap-server kafka-headless:9092
kafka-topics.sh --list --bootstrap-server kafka-headless:9092
kafka-console-producer.sh --broker-list kafka-headless:9092 --topic NetV2
>csv,separated,data
```

### Submitting Application to Spark
```bash
# Training a model
kubectl exec -ti spark-worker-0 -- spark-submit apps/train-model.py cross-validator --folds 2 -p 5 -s "setups/<setup_file>" -d "datasets/<dataset_file>" -o "models/<model_file>"

# Predictions
kubectl exec -ti spark-worker-0 -- spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 apps/kafka-predictions.py -b kafka-broker-0.kafka-headless.default.svc.cluster.local:9092 -t NetV2 --model "models/<model_file>" --schema schemas/<schema_file>
```
