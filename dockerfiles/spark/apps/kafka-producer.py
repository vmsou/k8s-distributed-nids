#!/usr/bin/env python

import argparse
import json
import os
import sys
import time

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType


def parse_arguments():
    # kafka-producer.py -b kafka-headless:9092 --schema "schemas/NetV2_schema.json" -t NetV2 -d "datasets/NF-UNSW-NB15-v2.parquet" --rows 10 --delay 5 -n 10
    parser = argparse.ArgumentParser(description="CSV Producer")
    parser.add_argument("-b", "--brokers", nargs="+", help="kafka.bootstrap.servers: host1:port1 host2:port2 ...)", required=True)
    parser.add_argument("-t", "--topic", help="Topic name", required=True)
    parser.add_argument("-f", "--format", help="Message format (csv, json)", default="csv", choices=["csv", "json"])
    parser.add_argument("-d", "--dataset", help="Data Source", required=True)
    parser.add_argument("--label", help="Label column name", default="Label")
    parser.add_argument("--schema", help="Path to Schema JSON", required=True)
    parser.add_argument("--rows", help="Number of rows to be sent in each batch", default=0, type=int)
    parser.add_argument("--delay", help="Seconds delay between batches", default=1, type=int)
    parser.add_argument("-n", help="Number of times to send rows", default=1, type=int)
    parser.add_argument("--seed", help="Seed", default=42, type=int)
    return parser.parse_args()


def create_session():
    name = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    spark: SparkSession = SparkSession \
        .builder \
        .appName(name) \
        .config("spark.sql.debug.maxToStringFields", '100') \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
        .getOrCreate()
    return spark


def get_stream_df(spark: SparkSession, path: str, schema: StructType):
    dataset_ext = os.path.basename(path).split(".")[-1]
    if dataset_ext == "csv": return spark.readStream.schema(schema).csv(path)
    elif dataset_ext == "parquet": return spark.readStream.schema(schema).parquet(path)
    else: raise Exception(f"File type: '{dataset_ext}' not supported for file '{path}'.")


def spark_schema_from_json(spark: SparkSession, path: str) -> StructType:
    schema_json = json.loads(spark.read.text(path).first()[0])
    return StructType.fromJson({"fields": schema_json["fields"]})


def main():
    args = parse_arguments()
    DATASET_PATH: str = args.dataset
    LABEL: str = args.label
    SCHEMA_PATH: str = args.schema
    BROKERS: list = ",".join(args.brokers)
    TOPIC: str = args.topic
    FORMAT = args.format
    ROWS: int = args.rows
    DELAY: int = args.delay
    N: int = args.n
    SEED: int = args.seed

    print("[ CONF ]".center(50, "-"))
    print(f"{DATASET_PATH=}")
    print(f"{LABEL=}")
    print(f"{SCHEMA_PATH=}")
    print(f"{BROKERS=}")
    print(f"{TOPIC=}")
    print(f"{FORMAT=}")
    print(f"{ROWS=}")
    print(f"{DELAY=}")
    print(f"{N=}")
    print()

    spark = create_session()

    print(" [SCHEMA] ".center(50, "-"))
    print(f"Loading {SCHEMA_PATH}...")

    t0 = time.time()
    schema = spark_schema_from_json(spark, SCHEMA_PATH)
    t1 = time.time()

    print(schema.simpleString())
    print(f"OK. Loaded in {t1 - t0}s")
    print()

    print(" [DATASET STREAM] ".center(50, "-"))
    t0 = time.time()
    df = get_stream_df(spark, DATASET_PATH, schema)
    t1 = time.time()
    print(f"OK. Loaded in {t1 - t0}s")
    print()

    format_expr: str = None
    if FORMAT == "csv":
        format_expr = "CAST(concat_ws(',', *) AS STRING) AS value"
    elif FORMAT == "json":
        format_expr = "to_json(struct(*)) AS value"
    else:
        raise Exception(f"Message format not supported: {FORMAT}")

    total_rows_to_send = ROWS * N

    def write_rows(batch_df: DataFrame, batch_id: int):
        collected_rows = batch_df.limit(total_rows_to_send).collect()
        total_collected = len(collected_rows)
        
        print(f"Total rows collected: {total_collected}")

        for i in range(N):
            start_idx = i * ROWS
            end_idx = min((i + 1) * ROWS, total_collected)

            if start_idx >= total_collected:
                break

            batch_rows = collected_rows[start_idx:end_idx]

            if batch_rows:
                batch_df = spark.createDataFrame(batch_rows, schema=schema)
                positives = batch_df.filter(F.col(LABEL) == 1).count()
                batch_df.selectExpr(format_expr) \
                    .write \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", BROKERS) \
                    .option("topic", TOPIC) \
                    .save()
                print(f"Batch {i + 1}: Sent {len(batch_rows)} rows with {positives} Positive(s).")

            time.sleep(DELAY)

    # rand_df = df.orderBy(F.rand(SEED))
    
    print(" [KAFKA] ".center(50, "-"))
    query = df \
        .writeStream \
        .queryName("Kafka Row Writer") \
        .foreachBatch(write_rows) \
        .outputMode("append") \
        .trigger(once=True) \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()
