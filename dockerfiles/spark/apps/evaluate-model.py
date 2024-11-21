import argparse
import json
import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.ml import PipelineModel
from pyspark.ml.base import Model
from pyspark.ml.evaluation import  BinaryClassificationEvaluator

from ensemble import EnsembleClassifier


def parse_arguments():
    # evaluate-model.py --test-ratio 0.7 -d "datasets/NF-UNSW-NB15-v2.parquet" "datasets/NF-ToN-IoT-v2.parquet" "datasets/NF-BoT-IoT-v2.parquet" "datasets/NF-CSE-CIC-IDS2018-v2.parquet" "datasets/NF-UQ-NIDS-v2.parquet" --model "models/GBT_PCA10_5F_PR_AUC_US-NF.model" "models/RF_PCA10_10F_PR_AUC_US-NF.model" "models/DTC_10F_PR_AUC_US-NF.model" --ensemble majority --output "metrics/ensemble-metrics.csv"
    parser = argparse.ArgumentParser(prog="evaluate-model", description="Evaluates a Model. Expects a PipelineModel")
    parser.add_argument("--schema", help="Path to Schema JSON", required=False)
    parser.add_argument("-d", "--dataset", nargs="+", help="Path(s) to Dataset(s)", required=True)
    parser.add_argument("--test-ratio", type=float, help="Ratio for test (0.0 to 1.0)", default=1.0, required=False)
    parser.add_argument("-m", "--model", nargs="+", help="Path(s) to Model(s)", required=True)
    parser.add_argument("--ensemble", help="Set ensemble mode (majority, attack, normal)", choices=["majority", "attack", "normal", None], default=None)
    parser.add_argument("-o", "--output", help="Path to Output (CSV)", default=None, required=False)
    parser.add_argument("--seed", type=float, help="Seed Number", default=42, required=False)

    return parser.parse_args()
        

def spark_schema_from_json(spark: SparkSession, path: str) -> StructType:
    schema_json = json.loads(spark.read.text(path).first()[0])
    return StructType.fromJson(schema_json)


def create_session():
    name = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    spark: SparkSession = SparkSession.builder \
        .appName(name) \
        .config("spark.sql.debug.maxToStringFields", '100') \
        .getOrCreate()
    return spark

def binary_metrics(predictions, target_col, prediction_col):
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
    print(f"{tp=}")
    print(f"{tn=}")
    print(f"{fp=}")
    print(f"{fn=}")
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
   
    # evaluator = BinaryClassificationEvaluator(labelCol=target_col)

    #areaUnderROC = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    #areaUnderPR = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})

    # return accuracy, precision, recall, f1_measure, tnr, areaUnderROC, areaUnderPR
    return tp, tn, fp, fn, accuracy, precision, recall, f1_measure #, areaUnderROC, areaUnderPR


def main():    
    args = parse_arguments()
    SCHEMA_PATH = args.schema
    DATASETS_PATHS = args.dataset
    MODELS_PATHS = args.model
    TEST_RATIO = args.test_ratio
    OUTPUT_PATH = args.output
    ENSEMBLE = args.ensemble
    SEED = args.seed

    print(" [CONF] ".center(50, "-"))
    print(f"{SCHEMA_PATH=}")
    print(f"{DATASETS_PATHS=}")
    print(f"{MODELS_PATHS=}")
    print(f"{TEST_RATIO=}")
    print(f"{OUTPUT_PATH=}")
    print(f"{ENSEMBLE=}")
    print()

    spark = create_session()

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

    # results_df = spark.createDataFrame([], schema="Dataset STRING, Model STRING, Accuracy DOUBLE, Precision DOUBLE, Recall DOUBLE, F1 DOUBLE, TNR DOUBLE, areaUnderROC DOUBLE, areaUnderPR DOUBLE")
    results_df = spark.createDataFrame([], schema="Dataset STRING, Model STRING, tp INTEGER, tn INTEGER, fp INTEGER, fn INTEGER, Accuracy DOUBLE, Precision DOUBLE, Recall DOUBLE, F1 DOUBLE")
    if ENSEMBLE:
        print(" [MODEL] ".center(50, "-"))
        print(f"Doing ensemble to {MODELS_PATHS}...")
        model: Model = EnsembleClassifier([PipelineModel.load(path) for path in MODELS_PATHS], mode=ENSEMBLE)
        model_name = f"{ENSEMBLE}(" + "/".join(os.path.basename(path) for path in MODELS_PATHS) + ")"
        target_col = model.getLabelCol()
        prediction_col = model.getPredictionCol()

        print("TARGET:", target_col)
        print("PREDICTIONS:", prediction_col)
        print()

        for dataset_path in DATASETS_PATHS:
            print(f" [METRICS] ".rjust(25, "-"))
            dataset_name = os.path.basename(dataset_path)
            print(f"Loading {dataset_name}...")
            t0 = time.time()
            df = spark.read.schema(schema).parquet(dataset_path) if schema else spark.read.parquet(dataset_path)
            t1 = time.time()
            print(f"OK. Done in {t1 - t0}s")
            print()
            
            if TEST_RATIO < 1.0:
                print("Splitting data...")
                df = df.sample(TEST_RATIO, seed=SEED)

            print("Making predictions...")
            predictions = model.transform(df)

            print("Calculating metrics...")
            t0 = time.time()
            # accuracy, precision, recall, f1_measure, tnr, areaUnderROC, areaUnderPR = binary_metrics(predictions, target_col, prediction_col)
            tp, tn, fp, fn, accuracy, precision, recall, f1_measure = binary_metrics(predictions, target_col, prediction_col)
            t1 = time.time()
            print(f"OK. Done in {t1 - t0}s")
            print()

            # result_df = spark.createDataFrame([(dataset_name, model_name, accuracy, precision, recall, f1_measure, tnr, areaUnderROC, areaUnderPR)])
            result_df = spark.createDataFrame([(dataset_name, model_name, tp, tn, fp, fn, accuracy, precision, recall, f1_measure)])
            results_df = results_df.union(result_df)
    else:
        for dataset_path in DATASETS_PATHS:
            print(" [DATASET] ".center(50, "-"))
            dataset_name = os.path.basename(dataset_path)
            # print(f" [{dataset_name}] ".center(70, "-"))
            print(f"Loading {dataset_name}...")
            t0 = time.time()
            df = spark.read.schema(schema).parquet(dataset_path) if schema else spark.read.parquet(dataset_path)
            t1 = time.time()
            print(f"OK. Done in {t1 - t0}s")
            print()
            
            if TEST_RATIO < 1.0:
                print("Splitting data...")
                df = df.sample(TEST_RATIO, seed=SEED)
            
            for model_path in MODELS_PATHS:
                print(" [MODEL] ".rjust(25, "-"))
                model_name = os.path.basename(model_path)
                print(f" [{model_name}] ".center(70, "-"))
                print(f"Loading {model_name}...")
                t0 = time.time()
                model: Model = PipelineModel.load(model_path)
                t1 = time.time()
                print(f"OK. Done in {t1 - t0}s")
                target_col = model.stages[-1].getLabelCol()
                prediction_col = model.stages[-1].getPredictionCol()
                
                print("TARGET:", target_col)
                print("PREDICTIONS:", prediction_col)
                print()

                print("Making predictions...")
                predictions = model.transform(df)
                print("Calculating metrics...")
                t0 = time.time()
                # accuracy, precision, recall, f1_measure, tnr, areaUnderROC, areaUnderPR = binary_metrics(predictions, target_col, prediction_col)
                tp, tn, fp, fn, accuracy, precision, recall, f1_measure = binary_metrics(predictions, target_col, prediction_col)
                t1 = time.time()
                print(f"{accuracy=}")
                print(f"{precision=}")
                print(f"{recall=}")
                print(f"{f1_measure=}")
                # print(f"{tnr=}")
                # print(f"{areaUnderROC=}")
                # print(f"{areaUnderPR=}")
                print(f"{tp=}")
                print(f"{tn=}")
                print(f"{fp=}")
                print(f"{fn=}")
                print(f"OK. Done in {t1 - t0}s")
                print()

                # result_df = spark.createDataFrame([(dataset_name, model_name, accuracy, precision, recall, f1_measure, tnr, areaUnderROC, areaUnderPR)])
                result_df = spark.createDataFrame([(dataset_name, model_name, tp, tn, fp, fn, accuracy, precision, recall, f1_measure)])
                results_df = results_df.union(result_df)

    print("OK. All models have been tested.")
    print()

    print(" [RESULTS] ".center(50, "-"))
    results_df.show()
    if OUTPUT_PATH:
        print(f"Saving results to {OUTPUT_PATH}...")
        t0 = time.time()
        results_df.write.csv(OUTPUT_PATH, header=True)
        t1 = time.time()
        print(f"OK. Done in {t1 - t0}s")

if __name__ == "__main__":
    main()


