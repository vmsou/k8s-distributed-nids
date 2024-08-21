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
    # kafka-predictions.py -b kafka:9092 -t NetV2 -m models/DTC_NETV2_MODEL --schema schemas/NetV2_schema.json
    parser = argparse.ArgumentParser(description="KafkaPredictions")
    parser.add_argument("-b", "--brokers", nargs="+", help="kafka.bootstrap.servers (i.e. <ip1>:<host1> <ip2>:<host2> ... <ipN>:<hostN>)", required=True)
    parser.add_argument("-t", "--topic", help="Kafka Topic (i.e. topic1)", required=True)
    parser.add_argument("-f", "--format", help="Format of data sent by topic", default="csv", choices=["csv", "json"])
    parser.add_argument("--schema", help="Path to Schema JSON", default="schemas/NetV2_schema.json", required=True)
    parser.add_argument("--model", help="Path to Model")
    return parser.parse_args()


def create_session() -> SparkSession:
    # .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    # .config("spark.cores.max", '1') \
    name = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    spark: SparkSession = SparkSession \
        .builder \
        .appName(name) \
        .config("spark.sql.debug.maxToStringFields", '100') \
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
    SCHEMA_PATH: str = args.schema
    FORMAT: str = args.format

    print()
    print(" [CONF] ".center(50, "-"))
    print("BROKERS:", BROKERS)
    print("TOPIC:", TOPIC)
    print("MODEL_PATH:", MODEL_PATH)
    print("SCHEMA_PATH:", SCHEMA_PATH)
    print("FORMAT:", FORMAT)
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
    target = model.stages[-1].getLabelCol()
    print("FEATURES COLUMN:", features_col)
    print("PREDICTION COLUMN:", prediction_col)
    print("TARGET:", target)

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
        .option("startingOffsets", "earliest") \
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

    query = predictions.writeStream \
        .queryName("Predictions Writer") \
        .format("console") \
        .outputMode("append") \
        .start()
    query.awaitTermination()


if __name__ == "__main__":
    main()
