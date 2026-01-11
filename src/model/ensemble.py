from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit
import json,os

class Ensemble:

    def __init__(self, models : dict):
        self.models : dict[str, tuple[PipelineModel, float]] = models
        self.metrics : dict = None
        self.total_weights : int = 0 
        for name, model in models.items():
            pipe = self._load(path=model)
            weight = self._get_recall(name)
            self.models[name] = (pipe, weight)
            self.total_weights += weight
        self.thresholds = self.total_weights / 2

    def _get_recall(self,model:str) -> float:
        if not self.metrics:
            with open(os.getenv("SAVE_MODEL_PATH") +"/metrics.json", "r") as f:
                self.metrics = json.load(f)
            f.close()

        return self.metrics[model]["recall"]

    def _load(self,path:str) -> PipelineModel:
        return PipelineModel.load(path)

    def classify(self,raw_df: DataFrame) -> DataFrame:
        ensemble_df = raw_df
        
        weighted_sum_expression = lit(0)

        for name, (model, weight) in self.models.items():
            pred_col = f"pred_{name}"
            predictions = model.transform(raw_df) \
                               .select("Patient ID", col("prediction").alias(pred_col))
            
            ensemble_df = ensemble_df.join(predictions, on="Patient ID", how="inner")
            
            weighted_sum_expression += (col(pred_col) * weight)

        ensemble_df = ensemble_df.withColumn("weighted_score", weighted_sum_expression)
        
        final_df = ensemble_df.withColumn(
            "ensemble_prediction",
            when(col("weighted_score") > self.thresholds, 1.0).otherwise(0.0)
        )

        return final_df