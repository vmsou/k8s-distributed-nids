from functools import reduce
from typing import List, Literal
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


class EnsembleClassifier(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    # string parameters as Spark Params
    predictionCol = Param(Params._dummy(), "predictionCol", "Column name for predictions")
    mode = Param(Params._dummy(), "mode", "Voting mode: majority, attack, or normal")

    def __init__(self, models: List[Transformer] = None, predictionCol: str = "ensemble_prediction", mode: Literal["majority", "attack", "normal"] = "majority"):
        super(EnsembleClassifier, self).__init__()
        self._setDefault(predictionCol="ensemble_prediction", mode="majority")

        self.models = models if models else []

        # Set the string-based Params
        self._set(predictionCol=predictionCol, mode=mode)

        self.labelCol = self._find_common_label_col()

        # unique prediction column for each model in the ensemble
        for i, model in enumerate(self.models):
            model.stages[-1].setPredictionCol(f"prediction_{i}")

    def _find_common_label_col(self):
        from collections import Counter
        label_cols = [model.stages[-1].getLabelCol() for model in self.models]
        return Counter(label_cols).most_common(1)[0][0]

    def getPredictionCol(self):
        return self.getOrDefault(self.predictionCol)

    def getLabelCol(self):
        return self.labelCol

    def _transform(self, df: DataFrame) -> DataFrame:
        if not self.models: raise ValueError("No models provided for ensemble voting.")

        df = df.withColumn("id", F.monotonically_increasing_id()).cache()

        # Collect predictions from each model
        predictions = [
            model.transform(df).select("id", f"prediction_{i}").cache()
            for i, model in enumerate(self.models)
        ]

        df = reduce(lambda df1, df2: df1.join(df2, on="id", how="inner"), [df] + predictions)

        prediction_cols = [F.col(f"prediction_{i}") for i in range(len(self.models))]
        mode = self.getOrDefault("mode")

        if mode == "majority": # Majority voting
            sum_predictions = reduce(lambda x, y: x + y, prediction_cols)
            majority_threshold = len(self.models) / 2
            df = df.withColumn(self.getOrDefault("predictionCol"), (sum_predictions > majority_threshold).cast("double"))
        elif mode == "attack": # At least one positive vote; then positive
            positive_vote = reduce(lambda x, y: x | y, [col == 1 for col in prediction_cols])
            df = df.withColumn(self.getOrDefault("predictionCol"), positive_vote.cast("double"))
        elif mode == "normal": # At least one negative vote; then negative
            negative_vote = reduce(lambda x, y: x | y, [col == 0 for col in prediction_cols])
            df = df.withColumn(self.getOrDefault("predictionCol"), (~negative_vote).cast("double"))
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'majority', 'attack', or 'normal'.")

        # Drop intermediate ID column
        df = df.drop("id")

        return df

    def copy(self, extra=None):
        """Creates a copy of this instance with the same params."""
        copied = EnsembleClassifier(self.models, self.getOrDefault("predictionCol"), self.getOrDefault("mode"))
        return self._copyValues(copied, extra)

    def save(self, path: str):
        """Save the model using Spark MLlib's save format"""
        raise NotImplementedError("Model saving is not fully implemented.")

    def fit(self, dataset: DataFrame) -> "EnsembleClassifier":
        """Fit method returns self to conform with the Estimator-Transformer interface."""
        return self
