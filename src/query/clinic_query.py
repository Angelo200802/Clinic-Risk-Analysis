from fastapi import APIRouter, HTTPException
from spark_manager import load_dataset
from pyspark.sql import DataFrame, functions as F
import os, logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

router_clinic_query = APIRouter()

ds : DataFrame = load_dataset(os.getenv("DATASET_PATH"))


derived_index = (
        ds
        .withColumn(
            "ShockIndex",
              F.col("Heart Rate") / F.col("Systolic Blood Pressure")
        )
        .withColumn(
            "ModifiedShockIndex", 
            F.col("Heart Rate") / F.col("Derived_MAP")
        )
        .withColumn(
            "AgeShockIndex", 
            F.col("ShockIndex") * F.col("Age")
        )
        .withColumn(
            "DiastolicShockIndex", 
            F.col("Heart Rate") / F.col("Diastolic Blood Pressure")
        )
        .withColumn(
            "PulsePressureIndex", 
            F.col("Derived_Pulse_Pressure") / F.col("Heart Rate")
        )
        .withColumn(
            "Rate_Pressure_Product", 
            F.col("Heart Rate") * F.col("Systolic Blood Pressure")
        )
        .withColumn(
            "Cardiac_Effort", 
            F.col("Heart Rate") * F.col("Derived_BMI")
        )
    )

avg_index = derived_index.agg( 
    F.avg("ShockIndex").alias("Avg_ShockIndex"),
    F.avg("ModifiedShockIndex").alias("Avg_ModifiedShockIndex"),
    F.avg("AgeShockIndex").alias("Avg_AgeShockIndex"),
    F.avg("DiastolicShockIndex").alias("Avg_DiastolicShockIndex"),
    F.avg("PulsePressureIndex").alias("Avg_PulsePressureIndex"),
    F.avg("Rate_Pressure_Product").alias("Avg_Rate_Pressure_Product"),
    F.avg("Cardiac_Effort").alias("Avg_Cardiac_Effort")
).toPandas().to_dict(orient="records")

@router_clinic_query.get("/clinic/derived_indices")
def get_derived_indices():
    return {
        "data": avg_index.toPandas().to_dict(orient="records")
    }

@router_clinic_query.get("/clinic/metabolic_bmi")
def get_metabolic_bmi():
    metabolic_bmi = (
        derived_index
        .select(
            "Derived_BMI",
            "Cardiac_Effort",
            "Risk Category"
        )
        .sample(withReplacement=False, fraction=0.05, seed=42)
    )
    return {
        "data": metabolic_bmi.toPandas().to_dict(orient="records")
    }