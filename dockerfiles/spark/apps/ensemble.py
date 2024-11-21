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
        self._setDefault(predictionCol=predictionCol, mode=mode)

        self.models = models if models else []

        # Set the string-based Params
        self._set(predictionCol=predictionCol, mode=mode)

        self.labelCol = models[0].stages[-1].getLabelCol()

    def getPredictionCol(self):
        return self.getOrDefault(self.predictionCol)

    def getLabelCol(self):
        return self.labelCol

    def _transform(self, df: DataFrame) -> DataFrame:
        if not self.models: raise ValueError("No models provided for ensemble voting.")

        predictions_col = "predictions"
        relevant_columns = df.columns + [predictions_col]

        df = df.withColumn(predictions_col, F.array())
        for model in self.models:
            prediction_col = model.stages[-1].getPredictionCol()
            df = model.transform(df)
            # Add prediction to array
            df = df.withColumn(predictions_col, F.concat(df[predictions_col], F.array(df[prediction_col])))
            df = df.select(*relevant_columns)

        mode = self.getOrDefault("mode")
        if mode == "majority": # Majority voting
            sum_predictions = F.expr(f"aggregate({predictions_col}, cast(0.0 as double), (acc, x) -> acc + x)")
            majority_threshold = len(self.models) / 2
            df = df.withColumn(self.getOrDefault("predictionCol"), (sum_predictions > majority_threshold).cast("double"))
        elif mode == "attack": # At least one positive vote; then positive
            positive_vote = F.expr(f"array_contains({predictions_col}, 1)")
            df = df.withColumn(self.getOrDefault("predictionCol"), positive_vote.cast("double"))
        elif mode == "normal": # At least one negative vote; then negative
            negative_vote = ~F.expr(f"array_contains({predictions_col}, 0)")
            df = df.withColumn(self.getOrDefault("predictionCol"), negative_vote.cast("double"))
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'majority', 'attack', or 'normal'.")

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
