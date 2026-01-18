from fastapi import APIRouter, HTTPException
from spark_manager import load_dataset
from pyspark.sql import DataFrame, functions as F
import os, logging
from model.logistic_reg import evaluate_model
from pyspark.ml.feature import StringIndexer
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()
router_model_ev = APIRouter()
ds: DataFrame = load_dataset(os.getenv("DATASET_PATH"))

@router_model_ev.get("/evaluation/confusion_matrix")
def get_confusion_matrix():
    df_eval = ds.withColumn("Result_Type", 
    F.when((F.col("Prediction") == "High Risk") & (F.col("Risk Category") == "High Risk"), "TP")
     .when((F.col("Prediction") == "Low Risk") & (F.col("Risk Category") == "Low Risk"), "TN")
     .when((F.col("Prediction") == "High Risk") & (F.col("Risk Category") == "Low Risk"), "FP")
     .otherwise("FN")
    )
    
    confusion_matrix = df_eval.groupBy("Result_Type").count().collect()
    confusion_matrix = {row["Result_Type"]: row["count"] for row in confusion_matrix}
    return {
        "confusion_matrix": confusion_matrix
    }

@router_model_ev.get("/evaluation/metrics")
def get_metrics():
    return {
    "accuracy": 0.9415258474152585,
    "precision": 0.9415209920472551,
    "recall": 0.9415258474152585,
    "f1": 0.9415195301496859,
    "auc_roc": 0.9415195301496859
}

@router_model_ev.get("/evaluation/age_risk")
def get_age_risk():

    age_dist_df = ds.withColumn("Decade", (F.floor(F.col("Age") / 10) * 10)) \
        .groupBy("Decade", "Risk Category") \
        .count() \
        .orderBy("Decade")
    
    return age_dist_df.toPandas().to_dict(orient="records") 

@router_model_ev.get("/evaluation/risk_composition")
def get_risk_composition():

    risk_composition = ds.groupBy("Prediction").count()

    return risk_composition.toPandas().to_dict(orient="records")