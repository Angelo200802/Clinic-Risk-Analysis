from fastapi import APIRouter, HTTPException
from spark_manager import load_dataset
from pyspark.sql import DataFrame, functions as F
import os, logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()
router_stats = APIRouter()
ds: DataFrame = load_dataset(os.getenv("DATASET_PATH"))

def get_columns():
    col_dict : dict = {}
    for col in ds.columns:
        col_dict[col.replace(" ","_").lower()] = col
    return col_dict

vital_signs = get_columns()


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

@router_stats.get("/stats")
def get_stats(signs:str):
    results = get_column_stats(ds, signs)
    
    if "error" in results:
        raise HTTPException(status_code=404, detail=results["error"])
    
    return results

@router_stats.get("/stats/summary")
def get_summary_stats():
    comparison_stats = ds\
        .groupBy("Prediction")\
        .agg(
            F.avg("Heart Rate").alias("Avg_HR"),
            F.avg("Systolic Blood Pressure").alias("Avg_SBP"),
            F.avg("Derived_Pulse_Pressure").alias("Avg_DPP")
        )
    return comparison_stats.toPandas().to_dict(orient="records")    

@router_stats.get("/stats/anomaly")
def get_anomaly_stats():
    outliers_count = ds.select(
        F.count(F.when(F.col("Body Temperature") > 40, 1)).alias("High_Fever_Cases"),
        F.count(F.when(F.col("Body Temperature") < 35, 1)).alias("Hypothermia_Cases"),
        F.count(F.when(F.col("Oxygen Saturation") < 90, 1)).alias("Hypoxia_Cases"),
        F.count(F.when(F.col("Heart Rate") > 150, 1)).alias("Tachycardia_Cases"),
        F.count(F.when(F.col("Heart Rate") < 50, 1)).alias("Bradycardia_Cases")
    )

    return outliers_count.toPandas().to_dict(orient="records")[0]