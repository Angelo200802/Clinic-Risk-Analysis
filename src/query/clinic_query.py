from fastapi import APIRouter, HTTPException
from spark_manager import get_session, load_dataset
from pyspark.sql import DataFrame, functions as F
import os, logging
from dotenv import load_dotenv

load_dotenv()

SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

router = APIRouter()

spark = get_session()
ds : DataFrame = load_dataset(os.getenv("DATASET_PATH"))

vital_signs = {
    "heart_rate": "Heart Rate"
}


def get_column_stats(df:DataFrame,column_name: str):
    
    if column_name not in vital_signs.keys():
        return {"error": f"Column {column_name} does not exist in the DataFrame."}
    
    stats = df.select(
        F.count(F.col(vital_signs[column_name])).alias("count"),
        F.mean(F.col(vital_signs[column_name])).alias("mean"),
        F.stddev(F.col(vital_signs[column_name])).alias("stddev"),
        F.min(F.col(vital_signs[column_name])).alias("min"),
        F.max(F.col(vital_signs[column_name])).alias("max"),
    ).first()
    
    return {
        "count": stats["count"],
        "mean": stats["mean"],
        "stddev": stats["stddev"],
        "min": stats["min"],
        "max": stats["max"],
    }

@router.get("/stats/{signs}")
def get_stats(signs:str):
    results = get_column_stats(ds, signs)
    
    if "error" in results:
        raise HTTPException(status_code=404, detail=results["error"])
    
    return results

@router.get("/clinic/shockindex")
def get_shock_index(order: str = "desc"):
    shock_df = ds \
        .withColumn("ShockIndex", F.col("Heart Rate") / F.col("Systolic Blood Pressure")) \
        .filter(F.col("ShockIndex") > 0.85) \
        .orderBy(F.col("ShockIndex").asc() if order == "asc" else F.col("ShockIndex").desc() ) \
        .select("Patient ID","Heart Rate", "Systolic Blood Pressure", "ShockIndex")
    pd_shock = shock_df.toPandas()  
    logging.info(f"Number of patients with Shock Index > 0.9: {len(pd_shock)}") 
    return {
        "data": pd_shock.head(5).to_dict(orient="records"),
        "count": len(pd_shock),   
    }

@router.get("/clinic/news2")
def get_news2():

    news_df = ds.withColumn(
        "Score_RR", # Frequenza Respiratoria
        F.when((F.col("Respiratory Rate") >= 25) | (F.col("Respiratory Rate") <= 8), 3)
         .when(F.col("Respiratory Rate").between(21, 24), 2)
         .when(F.col("Respiratory Rate").between(9, 11), 1)
         .otherwise(0) # 12-20
    ).withColumn(
        "Score_SpO2", # Saturazione Ossigeno (Scala 1 standard)
        F.when(F.col("Oxygen Saturation") <= 91, 3)
         .when(F.col("Oxygen Saturation").between(92, 93), 2)
         .when(F.col("Oxygen Saturation").between(94, 95), 1)
         .otherwise(0) # >= 96
    ).withColumn(
        "Score_Temp", # Temperatura Corporea
        F.when(F.col("Body Temperature") <= 35.0, 3)
         .when(F.col("Body Temperature") >= 39.1, 2)
         .when((F.col("Body Temperature") <= 36.0) | (F.col("Body Temperature").between(38.1, 39.0)), 1)
         .otherwise(0) # 36.1 - 38.0
    ).withColumn(
        "Score_SBP", # Pressione Sistolica
        F.when((F.col("Systolic Blood Pressure") <= 90) | (F.col("Systolic Blood Pressure") >= 220), 3)
         .when(F.col("Systolic Blood Pressure").between(91, 100), 2)
         .when(F.col("Systolic Blood Pressure").between(101, 110), 1)
         .otherwise(0) # 111-219
    ).withColumn(
        "Score_HR", # Frequenza Cardiaca
        F.when((F.col("Heart Rate") >= 131) | (F.col("Heart Rate") <= 40), 3)
         .when(F.col("Heart Rate").between(111, 130), 2)
         .when((F.col("Heart Rate").between(41, 50)) | (F.col("Heart Rate").between(91, 110)), 1)
         .otherwise(0) # 51-90
    ).withColumn(
        "NEWS2_Total",
        F.col("Score_RR") + F.col("Score_SpO2") + F.col("Score_Temp") + 
        F.col("Score_SBP") + F.col("Score_HR")
    ).filter(F.col("NEWS2_Total") >= 5).select(
        "Patient ID", "Respiratory Rate", "Oxygen Saturation", "Body Temperature",
        "Systolic Blood Pressure", "Heart Rate", "NEWS2_Total", "Risk Category"
    )

    pd_news2 = news_df.toPandas()
    return {
        "data": pd_news2.head(10).to_dict(orient="records"),
        "count": len(pd_news2)
    }

@router.get("/clinic/differentialpressure")
def get_differential_pressure(order: str = "desc"):
    diff_pressure_df = ds \
        .withColumn("DifferentialPressure", F.col("Systolic Blood Pressure") - F.col("Diastolic Blood Pressure")) \
        .withColumn("Label", F.when(F.col("DifferentialPressure") > 60, "High").otherwise(F.when(F.col("DifferentialPressure") < 25, "Low").otherwise("Normal"))) \
        .orderBy(F.col("DifferentialPressure").asc() if order == "asc" else F.col("DifferentialPressure").desc() ) \
        .select("Patient ID","Systolic Blood Pressure", "Diastolic Blood Pressure", "DifferentialPressure", "Label")
    pd_diff_pressure = diff_pressure_df.toPandas()  
    logging.info(f"Number of patients analyzed for Differential Pressure: {len(pd_diff_pressure)}") 
    return {
        "data": pd_diff_pressure.head(10).to_dict(orient="records") 
    }

@router.get("/clinic/differentialpressure/filter/{label}")
def filter_differential_pressure(label: str, order: str = "desc"):
    label = label.lower()
    diff_pressure_df = ds \
        .withColumn("DifferentialPressure", F.col("Systolic Blood Pressure") - F.col("Diastolic Blood Pressure")) \
        .withColumn("Label", F.when(F.col("DifferentialPressure") > 60, "high").otherwise(F.when(F.col("DifferentialPressure") < 25, "low").otherwise("normal"))) \
        .filter(F.col("Label") == label) \
        .orderBy(F.col("DifferentialPressure").asc() if order == "asc" else F.col("DifferentialPressure").desc() ) \
        .select("Patient ID","Systolic Blood Pressure", "Diastolic Blood Pressure", "DifferentialPressure", "Label")
    pd_diff_pressure = diff_pressure_df.toPandas()
    return {
        "data": pd_diff_pressure.head(10).to_dict(orient="records"),
        "count": len(pd_diff_pressure)
    }

