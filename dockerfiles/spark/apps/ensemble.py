from functools import reduce
from typing import List
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T


class EnsembleVotingClassifier(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, models: List[Transformer] = None, predictionCol: str = "majority_prediction", use_weights: bool = False):
        super(EnsembleVotingClassifier, self).__init__()
        self.models = models if models else []
        self.predictionCol = predictionCol
        self.probabilityCol = "probability"
        self.rawPredictionCol = "rawPrediction"
        self.use_weights = use_weights
        self.labelCol = self._find_common_label_col()
        for i, model in enumerate(self.models):
            model.stages[-1].setPredictionCol(f"prediction_{i}")

    def _find_common_label_col(self):
        from collections import Counter
        label_cols = [model.stages[-1].getLabelCol() for model in self.models]
        return Counter(label_cols).most_common(1)[0][0]

    def getPredictionCol(self) -> str:
        return self.predictionCol

    def getLabelCol(self) -> str:
        return self.labelCol

    def _transform(self, df: DataFrame) -> DataFrame:
        if not self.models:
            raise ValueError("No models provided for ensemble voting.")

        df = df.withColumn("id", F.monotonically_increasing_id()).cache()

        predictions = [
            model.transform(df).select("id", f"prediction_{i}").cache()
            for i, model in enumerate(self.models)
        ]

        df = reduce(lambda df1, df2: df1.join(df2, on="id", how="inner"), [df] + predictions)
        
        prediction_cols = [F.col(f"prediction_{i}") for i in range(len(self.models))]
        sum_predictions = reduce(lambda x, y: x + y, prediction_cols)
        num_models = len(self.models)
        majority_threshold = num_models / 2

        df = df.withColumn(self.predictionCol, (sum_predictions > majority_threshold).cast("double")).drop("id")

        df.unpersist()

        return df
    