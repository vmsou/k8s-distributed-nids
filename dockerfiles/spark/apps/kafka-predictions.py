import argparse
import json
import os
import sys
import time

import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import StructType


def parse_arguments():
    # apps/kafka-predictions.py -b kafka-headless:9092 -t NetV2 --model "models/RF_PCA10_10F_PR_AUC_US-NF.model" --schema schemas/NetV2_schema.json 
    parser = argparse.ArgumentParser(description="KafkaPredictions")
    parser.add_argument("-b", "--brokers", nargs="+", help="kafka.bootstrap.servers (i.e. <ip1>:<host1> <ip2>:<host2> ... <ipN>:<hostN>)", required=True)
    parser.add_argument("-t", "--topic", help="Kafka Topic (i.e. topic1)", required=True)
    parser.add_argument("-f", "--format", help="Format of data sent by topic", default="csv", choices=["csv", "json"])
    parser.add_argument("--schema", help="Path to Schema JSON", default="schemas/NetV2_schema.json", required=True)
    parser.add_argument("--model", help="Path to Model")

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
    MODEL_PATH: str = args.model
    TRIGGER_TYPE: str = args.trigger
    TRIGGER_INTERVAL: str = args.trigger_interval
    SCHEMA_PATH: str = args.schema
    FORMAT: str = args.format

    print()
    print(" [CONF] ".center(50, "-"))
    print(f"{BROKERS=}")
    print(f"{TOPIC=}")
    print(f"{MODEL_PATH=}")
    print(f"{TRIGGER_TYPE=}")
    if TRIGGER_TYPE == "interval":
        print(f"{TRIGGER_INTERVAL=}")
    print(f"{SCHEMA_PATH=}")
    print(f"{FORMAT=}")
    print()

    spark = create_session()
    sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")

    print(" [MODEL] ".center(50, "-"))
    print(f"Loading '{MODEL_PATH}'...")

    t0 = time.time()
    model = PipelineModel.load(MODEL_PATH)
    t1 = time.time()

    features_col = model.stages[-1].getFeaturesCol()
    prediction_col = model.stages[-1].getPredictionCol()
    target_col = model.stages[-1].getLabelCol()
    print("FEATURES COLUMN:", features_col)
    print("PREDICTION COLUMN:", prediction_col)
    print("TARGET:", target_col)

    print(f"OK. Loaded in {t1 - t0}s")
    print()

    print(" [SCHEMA] ".center(50, "-"))
    print(f"Loading '{SCHEMA_PATH}'...")

    t0 = time.time()
    schema = spark_schema_from_json(spark, SCHEMA_PATH)
    t1 = time.time()
    conditions = get_conditions_from_schema(schema)

    # for field in schema.jsonValue()["fields"]:
        # print(f"{field['name']}: {field['type']}")
    print(schema.simpleString())

    print(f"OK. Loaded in {t1 - t0}s")
    print()

    print(" [KAFKA] ".center(50, "-"))
    print("Setting stream...")
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", BROKERS) \
        .option("subscribe", TOPIC) \
        .option("startingOffsets", "latest") \
        .load()
    print("OK")

    # .selectExpr(f"from_csv(value, '{schema.simpleString()}') AS features")
    format_func = None
    if FORMAT == "csv": format_func = F.from_csv("value", schema.simpleString())
    elif FORMAT == "json": format_func = F.from_json("value", schema)
    else: raise Exception(f"Format not supported: {FORMAT}.")

    features = df \
        .selectExpr("CAST(value AS STRING)") \
        .select(format_func.alias("features")) \
        .select("features.*")
    
    valid_features = features.filter(F.expr(" AND ".join([c._jc.toString() for c in conditions])))

    print(" [PREDICTIONS] ".center(50, "-"))
    predictions = model.transform(valid_features).select(features_col, prediction_col, "probability")

    def process_batch(batch_df, batch_id):
        t0 = time.time()
        num_rows = batch_df.count()
        print(f"Processing batch {batch_id}: {num_rows} rows")
        batch_df.show()
        t1 = time.time()
        batch_duration = t1 - t0
        if batch_duration > 0:
            ev_per_sec = num_rows / batch_duration
            print(f"Batch {batch_id}: Processed {num_rows} rows in {batch_duration:.2f} seconds ({ev_per_sec:.2f} EV/S)")
    
    # Default micro-batch
    query = predictions.writeStream \
        .foreachBatch(process_batch) \
        .outputMode("append") \

    if TRIGGER_TYPE == "interval": query = query.trigger(processingTime=TRIGGER_INTERVAL)
    elif TRIGGER_TYPE == "available-now": query = query.trigger(availableNow=True)

    query.start().awaitTermination()

if __name__ == "__main__":
    main()
