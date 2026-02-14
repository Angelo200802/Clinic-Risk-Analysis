from fastapi import APIRouter, HTTPException
from query.model_evaluation import add_bmi_category
from spark_manager import load_dataset
from pyspark.sql import DataFrame, functions as F
import os, logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()
router_stats = APIRouter()
ds: DataFrame = load_dataset(os.getenv("DATASET_PATH"))

risk_by_gender = (
    ds.groupBy("Gender", "Risk Category")
    .count()
    .toPandas().to_dict(orient="records")
)

riks_by_bmi = (
    add_bmi_category(ds)
    .groupBy("BMI_Category", "Risk Category")
    .count()
    .toPandas().to_dict(orient="records")
)

risk_by_age = (
    ds.withColumn(
        "Decade", 
        (F.floor(F.col("Age") / 10) * 10)
    ) 
    .groupBy("Decade", "Risk Category") 
    .count() 
    .orderBy("Decade")
    .toPandas().to_dict(orient="records")
) 


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
        .groupBy("Risk Category")\
        .agg(
            F.avg("Heart Rate").alias("Avg_HR"),
            F.avg("Systolic Blood Pressure").alias("Avg_SBP"),
            F.avg("Derived_Pulse_Pressure").alias("Avg_DPP"),
            F.avg("Derived_BMI").alias("Avg_BMI"),
        )
    return comparison_stats.toPandas().to_dict(orient="records")    


@router_stats.get("/stats/age_risk")
def get_age_risk():

    if risk_by_age is None:
        logging.error(f"Error processing age risk data")
        raise HTTPException(status_code=500, detail="Error during data processing")
    return risk_by_age

@router_stats.get("/stats/gender_risk")
def get_gender_risk():

    if risk_by_gender is None:
        logging.error(f"Error processing gender risk data")
        raise HTTPException(status_code=500, detail="Error during data processing")
    return risk_by_gender

@router_stats.get("/stats/bmi_risk")
def get_bmi_risk():

    if riks_by_bmi is None:
        logging.error(f"Error processing BMI risk data")
        raise HTTPException(status_code=500, detail="Error during data processing")
    return riks_by_bmi

@router_stats.get("/stats/risk_composition")
def get_risk_composition():

    risk_composition = ds.groupBy("Risk Category").count()

    return risk_composition.toPandas().to_dict(orient="records")