from fastapi import APIRouter, HTTPException
from spark_manager import load_dataset
from model.ensemble import Ensemble
from pyspark.sql import DataFrame, functions as F
import os, logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()
router_model_ev = APIRouter()
model = Ensemble.build()

logging.log(logging.INFO, "Starting classification using Ensemble model...")
ds: DataFrame = model.classify(load_dataset(os.getenv("DATASET_PATH")))
logging.log(logging.INFO, "Classification completed.")

@router_model_ev.get("/classify")
def classify():
    pandas_df = ds.toPandas()
    return {"data":pandas_df.head(5).to_dict(orient="records")}

@router_model_ev.get("/evaluation/false_negatives")
def get_false_negatives():
    false_negative = ds.filter(

    )