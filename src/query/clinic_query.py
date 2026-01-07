from src.main import app
from src.spark_manager import get_session, load_dataset
from pyspark.sql import DataFrame, functions as F
from src.model.ensemble import Ensemble
import os

SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

spark = get_session()
ds : DataFrame = load_dataset(os.getenv("DATASET_PATH"))
ensemble_model = Ensemble(models={
    "logistic_regression": SAVE_MODEL_PATH+"/log_reg_pipeline",
    "naive_bayes": SAVE_MODEL_PATH+"/nb_pipeline",
    "mlp" : SAVE_MODEL_PATH+"/ann_model_pipeline"
})


@app.get("/stats/{signs}")
def get_stats(signs:str):
    return {}


@app.get("/shockindex")
def get_shock_index():
    shock_df = ds.withColumn("ShockIndex", F.col("Heart_Rate") / F.col("Systolic_BP")) \
        .filter(F.col("ShockIndex") > 0.9) \
        .select("Patient ID","Heart_Rate", "Systolic_BP", "ShockIndex")
    return {
        "shock_index": shock_df.toPandas().to_dict(orient="records"),
        "count": shock_df.count()   
    }