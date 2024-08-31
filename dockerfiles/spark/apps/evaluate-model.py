import argparse
import json
import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.ml import PipelineModel
from pyspark.ml.base import Model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


def parse_arguments():
    # evaluate-model.py -m models/UNSW_DTC_CV_PR_AUC -d datasets/NF-UNSW-NB15-v2.parquet
    parser = argparse.ArgumentParser(prog="evaluate-model", description="Evaluates a Model. Expects a PipelineModel")
    parser.add_argument("--schema", help="Path to Schema JSON", required=False)
    parser.add_argument("-d", "--dataset", help="Path to Dataset", required=True)
    parser.add_argument("--test-ratio", type=float, help="Ratio for test (0.0 to 1.0)", default=1.0)
    parser.add_argument("--seed", type=float, help="Seed Number", default=42)
    parser.add_argument("-m", "--model", help="Path to Model")
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "f1", "truePositiveRateByLabel", "falsePositiveRateByLabel", "precisionByLabel", "recallByLabel"], choices=["f1", "accuracy", "weightedPrecision", "weightedRecall", "weightedTruePositiveRate", "weightedFalsePositiveRate", "weightedFMeasure", "truePositiveRateByLabel", "falsePositiveRateByLabel", "precisionByLabel", "recallByLabel", "fMeasureByLabel", "logLoss", "hammingLoss"])
    parser.add_argument("--metrics-label", type=float, default=1.0)

    return parser.parse_args()
        

def spark_schema_from_json(spark: SparkSession, path: str) -> StructType:
    schema_json = json.loads(spark.read.text(path).first()[0])
    return StructType.fromJson(schema_json)


def create_session():
    #    .config("spark.cores.max", '3') \
    name = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    spark: SparkSession = SparkSession.builder \
        .appName(name) \
        .config("spark.sql.debug.maxToStringFields", '100') \
        .getOrCreate()
    return spark

def show_binary_metrics(predictions, target_col, prediction_col, metrics):
    # areaUnderROC (balanced) | areaUnderPR (imbalanced)

    t0 = predictions[target_col] == 0
    t1 = predictions[target_col] == 1
    p0 = predictions[prediction_col] == 0
    p1 = predictions[prediction_col] == 1
    
    tp = predictions.filter(t1 & p1).count()
    tn = predictions.filter(t0 & p0).count()
    fp = predictions.filter(t0 & p1).count()
    fn = predictions.filter(t1 & p0).count()
    
    print("Confusion Matrix:")
    # print(f"{tp=}")
    # print(f"{tn=}")
    # print(f"{fp=}")
    # print(f"{fn=}")
    print()
    print(f"{'Actual/Predicted':^15} | {'Positive':^10} | {'Negative':^10}")
    print(f"{'-'*42}")
    print(f"{'Positive':^15} | {tp:^10} | {fn:^10}")
    print(f"{'Negative':^15} | {fp:^10} | {tn:^10}")
    print()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0  
    f1_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0  
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 measure: {f1_measure}")
    print(f"True Negative Rate: {tnr}")

    print()
    evaluator = BinaryClassificationEvaluator(labelCol=target_col)

    print("Binary Metrics:")
    for metric in metrics:
        score = evaluator.evaluate(predictions, {evaluator.metricName: metric})
        print(f"{metric}: {score}")


def show_multiclass_metrics(predictions, target_col, metrics, metricLabel=1.0):
    # f1|accuracy|weightedPrecision|weightedRecall|weightedTruePositiveRate|weightedFalsePositiveRate|weightedFMeasure|truePositiveRateByLabel|falsePositiveRateByLabel|precisionByLabel|recallByLabel|fMeasureByLabel|logLoss|hammingLoss

    evaluator = MulticlassClassificationEvaluator(labelCol=target_col, metricLabel=metricLabel)


    for metric in metrics:
        score = evaluator.evaluate(predictions, {evaluator.metricName: metric})
        print(f"{metric} (metricLabel={metricLabel}): {score}")


def main():    
    args = parse_arguments()
    SCHEMA_PATH = args.schema
    DATASET_PATH = args.dataset
    MODEL_PATH = args.model
    TEST_RATIO = args.test_ratio
    METRICS = args.metrics
    METRICS_LABEL = args.metrics_label
    SEED = args.seed

    print(" [CONF] ".center(50, "-"))
    print(f"{SCHEMA_PATH=}")
    print(f"{SCHEMA_PATH=}")
    print(f"{MODEL_PATH=}")
    print(f"{DATASET_PATH=}")
    print(f"{TEST_RATIO=}")
    print(f"{METRICS=}")
    print(f"{METRICS_LABEL=}")
    print()

    spark = create_session()

    print(" [MODEL] ".center(50, "-"))
    print(f"Loading {MODEL_PATH}...")
    t0 = time.time()
    model: Model = PipelineModel.load(MODEL_PATH)
    t1 = time.time()
    print(f"OK. Done in {t1 - t0}s")
    target_col = model.stages[-1].getLabelCol()
    features_col = model.stages[-1].getFeaturesCol()
    prediction_col = model.stages[-1].getPredictionCol()
        
    print("TARGET:", target_col)
    print("FEATURES:", features_col)
    print("PREDICTIONS:", prediction_col)
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
    print(f"Loading {DATASET_PATH}")
    t0 = time.time()
    df = spark.read.schema(schema).parquet(DATASET_PATH) if schema else spark.read.parquet(DATASET_PATH)
    t1 = time.time()
    print(f"OK. Done in {t1 - t0}s")
    print()

    print(f"Splitting data into TEST: {TEST_RATIO}")
    test_df = df.sample(TEST_RATIO, seed=SEED)
    print()

    print(" [PREDICTIONS] ".center(50, "-"))
    print("Making predictions...")
    t0 = time.time()
    predictions = model.transform(test_df)
    pred_labels = predictions.select(features_col, "probability", prediction_col, target_col)
    pred_labels.show(3)
    t1 = time.time()
    print(f"OK. Done in {t1 - t0}s")
    print()

    print(" [METRICS] ".center(50, "-"))
    t0 = time.time()
    show_binary_metrics(predictions, target_col, prediction_col, ["areaUnderROC", "areaUnderPR"])
    t1 = time.time()
    print(f"OK. Done in {t1 - t0}s")
    print()

    print("Multiclass Metrics:")
    t0 = time.time()
    show_multiclass_metrics(predictions, target_col, METRICS, METRICS_LABEL)
    t1 = time.time()
    print(f"OK. Done in {t1 - t0}s")
    print()


if __name__ == "__main__":
    main()


