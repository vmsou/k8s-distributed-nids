from typing import Counter
from pyspark.ml import Transformer
from pyspark.ml.base import Model
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql.functions import col, expr, monotonically_increasing_id, lit
from pyspark.sql import DataFrame


class EnsembleVotingClassifier(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, models=None, predictionCol: str = "majority_prediction"):
        super(EnsembleVotingClassifier, self).__init__()
        self.models = models if models else []
        # self.stages = [stage for model in models for stage in model.stages]
        self.predictionCol = predictionCol
        self.labelCol = Counter([model.stages[-1].getLabelCol() for model in models]).most_common(1)[0][0]
        for i, model in enumerate(models): model.stages[-1].setPredictionCol(f"prediction_{i}")

    def getPredictionCol(self) -> str: return self.predictionCol
    def getLabelCol(self) -> str: return self.labelCol
    
    def _transform(self, df: DataFrame) -> DataFrame:
        if not self.models: raise ValueError("No models provided for ensemble voting.")
        df = df.withColumn("id", monotonically_increasing_id())
        transformed_df = df
        predictions = [model.transform(df).select("id", f"prediction_{i}") for i, model in enumerate(self.models)]
        
        for pred in predictions:
            transformed_df = transformed_df.join(pred, "id", "outer")

        # Majority voting
        # num_models = len(self.models)
        # vote_array_expr = "array(" + ", ".join([f"prediction_{i}" for i in range(num_models)]) + ")"
        # vote_expr = f"double(array_sum({vote_array_expr}) / {num_models} >= 0.5) "
        # return dataset.join(combined_predictions.withColumn(self.predictionCol, expr(vote_expr)))

        # Majority voting
        num_models = len(self.models)
        vote_expr = f"double((" + " + ".join([f"prediction_{i}" for i in range(num_models)]) + f") / {num_models} >= 0.5)"
        return transformed_df.withColumn(self.predictionCol, expr(vote_expr))
