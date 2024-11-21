import argparse
import json
import os
import sys
import time

import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.base import Model
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import StructType

from ensemble import EnsembleClassifier


def parse_arguments():
    # apps/kafka-predictions.py -b kafka-headless:9092 -t NetV2 --model "models/RF_PCA10_10F_PR_AUC_US-NF.model" --schema schemas/NetV2_schema.json 
    parser = argparse.ArgumentParser(description="KafkaPredictions")
    parser.add_argument("-b", "--brokers", nargs="+", help="kafka.bootstrap.servers (i.e. <ip1>:<host1> <ip2>:<host2> ... <ipN>:<hostN>)", required=True)
    parser.add_argument("-t", "--topic", help="Kafka Topic (i.e. topic1)", required=True)
    parser.add_argument("-f", "--format", help="Format of data sent by topic", default="csv", choices=["csv", "json"])
    parser.add_argument("--schema", help="Path to Schema JSON", default="schemas/NetV2_schema.json", required=True)
    parser.add_argument("-m", "--model", nargs="+", help="Path(s) to Model(s). If more than one model, ensemble will be used.")
    parser.add_argument("--ensemble", help="Set ensemble mode (majority, attack, normal)", choices=["majority", "attack", "normal", None], default=None)


    parser.add_argument("--metric", help="Displays EV/S", action="store_true")
    parser.add_argument("--trigger", help="Type of trigger (micro-batch, interval, available-now)", choices=["micro-batch", "interval", "available-now"], default="micro-batch")
    parser.add_argument("--trigger-interval", help="Trigger interval time (e.g., '1 second', '10 seconds', '1 minute')", default="1 second")

    return parser.parse_args()



def create_session() -> SparkSession:
    name = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    spark: SparkSession = SparkSession \
        .builder \
        .appName(name) \
        .config("spark.sql.debug.maxToStringFields", '100') \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
        .getOrCreate()
    return spark


def spark_schema_from_json(spark: SparkSession, path: str) -> StructType:
    schema_json = json.loads(spark.read.text(path).first()[0])
    return StructType.fromJson({"fields": schema_json["fields"]})


def get_conditions_from_schema(schema: StructType) -> list[F.Column]:
    conditions = []
    for field in schema.fields:
        if not field.nullable: conditions.append(F.col(field.name).isNotNull())
    return conditions


def main() -> None:
    args = parse_arguments()
    BROKERS: str = ",".join(args.brokers)
    TOPIC: str = args.topic
    MODEL_PATH: list[str] = args.model
    ENSEMBLE = args.ensemble
    TRIGGER_TYPE: str = args.trigger
    TRIGGER_INTERVAL: str = args.trigger_interval
    SCHEMA_PATH: str = args.schema
    FORMAT: str = args.format
    METRIC: bool = args.metric

    print()
    print(" [CONF] ".center(50, "-"))
    print(f"{BROKERS=}")
    print(f"{TOPIC=}")
    print(f"{MODEL_PATH=}")
    print(f"{ENSEMBLE=}")

    print(f"{TRIGGER_TYPE=}")
    if TRIGGER_TYPE == "interval":
        print(f"{TRIGGER_INTERVAL=}")
    print(f"{SCHEMA_PATH=}")
    print(f"{FORMAT=}")
    print(f"{METRIC=}")
    print()

    spark = create_session()
    sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")

    print()
    print(" [MODEL] ".center(50, "-"))
    print(f"Loading {MODEL_PATH}...")

    if ENSEMBLE or len(MODEL_PATH) > 1:
        print("Doing ensemble...")
        t0 = time.time()
        model: Model = EnsembleClassifier([PipelineModel.load(path) for path in MODEL_PATH], mode=ENSEMBLE)
        t1 = time.time()
        print(f"OK. Done in {t1 - t0}s")
        target_col = model.getLabelCol()
        prediction_col = model.getPredictionCol()
    else:
        t0 = time.time()
        model = PipelineModel.load(MODEL_PATH[0])
        t1 = time.time()
        prediction_col = model.stages[-1].getPredictionCol()
        target_col = model.stages[-1].getLabelCol()
        print(f"OK. Loaded in {t1 - t0}s")

    print("PREDICTION COLUMN:", prediction_col)
    print("TARGET:", target_col)

    print()

    print(" [SCHEMA] ".center(50, "-"))
    print(f"Loading '{SCHEMA_PATH}'...")

    t0 = time.time()
    schema = spark_schema_from_json(spark, SCHEMA_PATH)
    t1 = time.time()
    conditions = get_conditions_from_schema(schema)

    # for field in schema.jsonValue()["fields"]:
    #   print(f"{field['name']}: {field['type']}")
    print(schema.simpleString())

    print(f"OK. Loaded in {t1 - t0}s")
    print()

    print(" [KAFKA] ".center(50, "-"))
    kafka_params = {
        "kafka.bootstrap.servers": BROKERS,
        "subscribe": TOPIC,
        "startingOffsets": "latest",
        #"max.partition.fetch.bytes": "52428800",  # 50 MB
        #"fetch.min.bytes": "1048576",  # 1 MB
        #"fetch.max.wait.ms": "100",    # 100 ms
        #"consumer.poll.timeout.ms": "500"
    }   
    print("Setting stream...")
    print(kafka_params)
    stream_df = spark.readStream \
        .format("kafka") \
        .options(**kafka_params) \
        .load()
    print(f"OK.")

    # .selectExpr(f"from_csv(value, '{schema.simpleString()}') AS features")
    format_func = None
    if FORMAT == "csv": format_func = F.from_csv("value", schema.simpleString())
    elif FORMAT == "json": format_func = F.from_json("value", schema)
    else: raise Exception(f"Format not supported: {FORMAT}.")

    #stream_df = stream_df.selectExpr("CAST(value AS STRING)")
    #parsed_df = stream_df.select(format_func.alias("features")).select("features.*").filter(F.expr(" AND ".join([c._jc.toString() for c in conditions])))
    parsed_df = stream_df.selectExpr("CAST(value AS STRING)", "partition", "offset", "timestamp") \
                     .select(format_func.alias("features"), "partition", "offset", "timestamp") \
                     .select("features.*", "partition", "offset", "timestamp")

    
    print(" [PREDICTIONS] ".center(50, "-"))
    predictions = model.transform(parsed_df)

    def process_batch_eps(batch_df, batch_id):
        t0 = time.time()
        event_count = batch_df.count()
        t1 = time.time()
        time_taken = t1 - t0
        if time_taken > 0:
            eps = event_count / time_taken
            print(f"Batch {batch_id}: {eps:.2f} EV/S - {event_count} events in {time_taken:.2f} seconds")
        else:
            print(f"Batch {batch_id}: No time taken for batch processing (likely empty batch).")
    
    def process_batch(batch_df, batch_id):
        print(f"Batch {batch_id} processed")
        batch_df.show()
    
    # Default micro-batch
    if METRIC:
        query = predictions.writeStream.foreachBatch(process_batch_eps)
    else:
        query = predictions.writeStream.foreachBatch(process_batch)

    if TRIGGER_TYPE == "interval": query = query.trigger(processingTime=TRIGGER_INTERVAL)
    elif TRIGGER_TYPE == "available-now": query = query.trigger(availableNow=True)
    
    query.outputMode("append").start().awaitTermination()

if __name__ == "__main__":
    main()
