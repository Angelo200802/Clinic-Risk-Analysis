from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit
from enum import Enum
import json,os, dotenv

dotenv.load_dotenv()

class Ensemble:

    class Weigth(Enum):
        RECALL = "recall"
        F1_SCORE = "f1"
        PRECISION = "precision"
        ACCURACY = "accuracy"

    def __init__(self,models : dict = None,weights_by: Weigth = Weigth.RECALL):
        self.models : dict[str, tuple[PipelineModel, float]] = {
            "logistic_regression": os.getenv("SAVE_MODEL_PATH") + "/log_reg_pipeline",
            "mlp": os.getenv("SAVE_MODEL_PATH") + "/ann_model_pipeline",
            "naive_bayes": os.getenv("SAVE_MODEL_PATH") + "/nb_pipeline"
        } if not models else models
        self.weights_by = weights_by.value
        self.metrics : dict = None
        self.total_weights : int = 0 
        for name, model in self.models.items():
            pipe = self._load(path=model)
            weight = self._get_weigths(name)
            self.models[name] = (pipe, weight)
            self.total_weights += weight
        self.thresholds = self.total_weights / 2

    def _get_weigths(self,model:str) -> float:
        if not self.metrics:
            with open(os.getenv("SAVE_MODEL_PATH") +"/metrics.json", "r") as f:
                self.metrics = json.load(f)
            f.close()

        return self.metrics[model][self.weights_by]

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
            "Prediction",
            when(col("weighted_score") > self.thresholds, "Low Risk").otherwise("High Risk")
        )

        return final_df
