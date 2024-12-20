import argparse
import json
import os
import sys
import time

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import Evaluator, MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import Row


Estimator_t = CrossValidator | Pipeline

def parse_arguments():
    # train-model.py cross-validator --folds 3 -p 5 -s "setups/SETUP_DTC_NETV2_MODEL_CV" -d "datasets/NF-UNSW-NB15-v2.parquet" -o "models/DTC_NETV2_MODEL2"
    parent_parser = argparse.ArgumentParser(prog="create-model", add_help=False)

    parent_parser.add_argument("-s", "--setup", help="Path to Setup Folder", required=True)
    parent_parser.add_argument("--schema", help="Path to Schema JSON", required=False)
    parent_parser.add_argument("-d", "--dataset", help="Path to Dataset", required=True)
    parent_parser.add_argument("--train-ratio", type=float, help="Ratio for training (0.0 to 1.0)", default=1.0)
    parent_parser.add_argument("--partitions", help="Dataset repartitions (defaults to sparkContext.defaultParallelism * 2)", required=False, default=None, type=int)
    parent_parser.add_argument("--seed", type=float, help="Seed Number", default=42)
    parent_parser.add_argument("-o", "--output", help="Path to Model Output")
    parent_parser.add_argument("--log", help="Path to Training Metrics CSV", default=None)

    main_parser = argparse.ArgumentParser()
    subparser = main_parser.add_subparsers(dest="command", required=True, help="Choose model")

    pipeline = subparser.add_parser("pipeline", help="Pipeline", parents=[parent_parser])
    
    cross_validator = subparser.add_parser("cross-validator", help="CrossValidator", parents=[parent_parser])
    cross_validator.add_argument("--metric", help="Metric name", required=False, default="accuracy", choices=["areaUnderROC", "areaUnderPR", "f1", "accuracy", "weightedPrecision", "weightedRecall", "weightedTruePositiveRate", "weightedFalsePositiveRate", "weightedFMeasure", "truePositiveRateByLabel", "falsePositiveRateByLabel", "precisionByLabel", "recallByLabel", "fMeasureByLabel", "logLoss", "hammingLoss"])
    cross_validator.add_argument("--metric-label", help="Metric label", required=False, default=1.0, type=float)
    cross_validator.add_argument("-p", "--parallelism", help="Parallelism's Number (defaults to sparkContext.defaultParallelism)", type=int, default=None)
    cross_validator.add_argument("-f", "--folds", help="Number of Folds. Must be over 1 fold", type=int, default=2)

    return main_parser.parse_args()


def create_session():
    name = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    spark: SparkSession = SparkSession.builder \
        .appName(name) \
        .config("spark.sql.debug.maxToStringFields", '100') \
        .getOrCreate()
    # .config("spark.sql.autoBroadcastJoinThreshold", 209715200) \  # 200MB
    
    return spark


def spark_schema_from_json(spark: SparkSession, path: str) -> StructType:
    schema_json = json.loads(spark.read.text(path).first()[0])
    return StructType.fromJson(schema_json)


def main():    
    args = parse_arguments()
    COMMAND = args.command
    SETUP_PATH = args.setup
    DATASET_PATH = args.dataset
    TRAIN_RATIO = args.train_ratio
    PARTITIONS = args.partitions
    SCHEMA_PATH = args.schema
    OUTPUT_PATH = args.output
    LOG_PATH = args.log
    SEED = args.seed
    
    print(" [CONF] ".center(50, "-"))
    print(f"{COMMAND=}")
    print(f"{SETUP_PATH=}")
    print(f"{SCHEMA_PATH=}")
    print(f"{DATASET_PATH=}")
    print(f"{TRAIN_RATIO=}")
    print(f"{PARTITIONS=}")
    print(f"{OUTPUT_PATH=}")
    print(f"{LOG_PATH=}")
    print()

    spark = create_session()


    print(" [SETUP] ".center(50, "-"))
    print(f"Loading {SETUP_PATH}...")
    t0 = time.time()
    if COMMAND == "cross-validator":
        if args.folds < 1: raise Exception(f"Folds must be >= 1, not: {args.folds}")
        METRIC= args.metric
        METRIC_LABEL = args.metric_label
        PARALLELISM = args.parallelism
        FOLDS = args.folds

        estimator: CrossValidator = CrossValidator.load(SETUP_PATH)
        if PARALLELISM is None:
            PARALLELISM = spark.sparkContext.defaultParallelism
            print(f"Parallelism is set to None. Defaulting to {PARALLELISM}")

        print(f"Setting evaluator...")
        LABEL_COL = estimator.getEstimator().getStages()[-1].getLabelCol()
        PREDICTION_COL = estimator.getEstimator().getStages()[-1].getPredictionCol()
        print(f"{METRIC=}")
        print(f"{METRIC_LABEL=}")
        print(f"{LABEL_COL=}")
        print(f"{PREDICTION_COL=}")

        if METRIC in ("areaUnderROC", "areaUnderPR"):
            evaluator: Evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName=METRIC)
        else:
            evaluator: Evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol=PREDICTION_COL, metricName=METRIC, metricLabel=METRIC_LABEL)

        estimator = estimator.setNumFolds(FOLDS).setParallelism(PARALLELISM).setEvaluator(evaluator)
    elif COMMAND == "pipeline":
        estimator: Pipeline = Pipeline.load(SETUP_PATH)
        LABEL_COL = estimator.getStages()[-1].getLabelCol()
        PREDICTION_COL = estimator.getStages()[-1].getPredictionCol()

    t1 = time.time()
    print(f"OK. Done in {t1 - t0}s")
    print()

    schema = None
    if SCHEMA_PATH:
        print(" [SCHEMA] ".center(50, "-"))
        print(f"Loading {SCHEMA_PATH}...")

        t0 = time.time()
        schema = spark_schema_from_json(spark, SCHEMA_PATH)
        t1 = time.time()

        # for field in schema.fields:  print(f"{field.name}: {field.typeName()}")
        print(schema.simpleString())
        print()

    print(" [DATASET] ".center(50, "-"))
    print(f"Loading {DATASET_PATH}...")
    t0 = time.time()
    train_df = spark.read.schema(schema).parquet(DATASET_PATH) if schema else spark.read.parquet(DATASET_PATH)
    t1 = time.time()
    print(f"OK. Done in {t1 - t0}s")
    print()

    print("Setting persist to MEMORY_AND_DISK...")
    train_df.persist(StorageLevel.MEMORY_AND_DISK)

    print(f"Splitting data into {TRAIN_RATIO} split...")
    train_df = train_df.sample(TRAIN_RATIO, seed=SEED)

    if PARTITIONS is None:
        PARTITIONS = spark.sparkContext.defaultParallelism * 2

    print(f"Repartitioning into {PARTITIONS} partitions...")

    train_df = train_df.repartition(PARTITIONS)

    print(" [MODEL] ".center(50, "-"))
    print("Training model...")
    t0 = time.time()
    model: Estimator_t = estimator.fit(train_df)
    t1 = time.time()
    train_time = t1 - t0
    if COMMAND == "cross-validator": model = model.bestModel
    print(f"OK. Done in {t1 - t0}s")
    print()
    print(model.stages[-1])
    print()

    print(f"Saving model to {OUTPUT_PATH}...")
    model.write().overwrite().save(OUTPUT_PATH)
    print("OK")

    if LOG_PATH:
        print(f"Saving training metrics to '{LOG_PATH}'...")
        row = Row(cores=spark.sparkContext.defaultParallelism, setup=os.path.basename(SETUP_PATH), dataset=os.path.basename(DATASET_PATH), model=os.path.basename(OUTPUT_PATH), train_ratio=TRAIN_RATIO, training_time=train_time)
        training_data = spark.createDataFrame([row])
        training_data.show()
        try:
            training_data.write.csv(path=LOG_PATH, mode="append", header=True)
            print(f"OK.")
        except Exception as e:
            print(f"Error writing to {LOG_PATH}: {e}")
    else:
        print("Log path not set. Not saving metrics to file.")


if __name__ == "__main__":
    main()
