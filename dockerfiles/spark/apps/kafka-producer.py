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
    # kafka-producer.py -d datasets/NF-UNSW-NB15-v2.parquet --schema schemas/NetV2_schema.json --topic NetV2 -n 10 --delay 1
    parser = argparse.ArgumentParser(description="CSV Producer")
    parser.add_argument("-b", "--brokers", nargs="+", help="kafka.bootstrap.servers: host1:port1 host2:port2 ...)", required=True)
    parser.add_argument("-t", "--topic", help="Topic name", required=True)
    parser.add_argument("-f", "--format", help="Message format (csv, json)", default="csv", choices=["csv", "json"])
    parser.add_argument("-d", "--dataset", help="Data Source", required=True)
    parser.add_argument("--schema", help="Path to Schema JSON", required=True)
    parser.add_argument("-n", help="Total number of rows (0 for all)", default=0, type=int)
    parser.add_argument("--delay", help="Seconds of delay for each row to be sent", default=1, type=int)
    parser.add_argument("--seed", help="Seed", default=42, type=int)
    return parser.parse_args()


def create_session():
    # .config("spark.cores.max", '1') \
    name = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    spark: SparkSession = SparkSession \
        .builder \
        .appName(name) \
        .config("spark.sql.debug.maxToStringFields", '100') \
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
    SCHEMA_PATH: str = args.schema
    BROKERS: list = ",".join(args.brokers)
    TOPIC: str = args.topic
    FORMAT = args.format
    N: int = args.n
    DELAY: int = args.delay
    SEED: int = args.seed
	
    print("[ CONF ]".center(50, "-"))
    print("DATASET_PATH:", DATASET_PATH)
    print("SCHEMA_PATH:", SCHEMA_PATH)
    print("SERVERS:", BROKERS)
    print("TOPIC:", TOPIC)
    print("FORMAT:", FORMAT)
    print("N:", N)
    print("DELAY:", DELAY)
    print()

    spark = create_session()        

    print(" [SCHEMA] ".center(50, "-"))
    print(f"Loading {SCHEMA_PATH}...")

    t0 = time.time()
    schema = spark_schema_from_json(spark, SCHEMA_PATH)
    t1 = time.time()

    #for field in schema.jsonValue()["fields"]:
    #    print(f"{field['name']}: {field['type']}")
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
    if FORMAT == "csv": format_expr = "CAST(concat_ws(',', *) AS STRING) AS value"
    elif FORMAT == "json": format_expr = "to_json(struct(*)) AS value"
    else: raise Exception(f"Message format not supported: {FORMAT}")

    batch_count = 0
    def write_row(batch_df: DataFrame, batch_id: int):
        nonlocal batch_count

        batch_count += 1
        rows = batch_df.collect()
        print(f"Batch: {batch_count}")
        for row in rows:
            print(",".join(map(str, row.asDict().values())), '\n')
            row_df = spark.createDataFrame([row], schema=schema)
            row_df.selectExpr(format_expr) \
                .write \
                .format("kafka") \
                .option("kafka.bootstrap.servers", BROKERS) \
                .option("topic", TOPIC) \
                .save()
            time.sleep(DELAY)

    rand_df = df.orderBy(F.rand(SEED))  # TODO: Fix - Not randomizing stream
    if N > 0: rand_df = df.limit(int(N))
    
    print(" [KAFKA] ".center(50, "-"))
    query = rand_df \
            .writeStream \
            .queryName("Kafka Row Writer") \
            .foreachBatch(write_row) \
            .outputMode("append") \
            .trigger(once=True) \
            .start()
    query.awaitTermination()


if __name__ == "__main__":
	main()
