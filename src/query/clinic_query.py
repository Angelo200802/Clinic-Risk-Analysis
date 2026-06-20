from fastapi import APIRouter
from spark_manager import load_dataset
from pyspark.sql import DataFrame, Window, functions as F
import os, logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

router_clinic_query = APIRouter()

ds : DataFrame = load_dataset(os.getenv("DATASET_PATH"))

_cache = {}

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
)
derived_index.cache() 

@router_clinic_query.get("/clinic/derived_indices")
def get_derived_indices():
    return derived_indices()

def derived_indices():
    
    if not 'avg_index' in _cache:
        _cache['avg_index'] = (
            derived_index
            .groupBy("Risk Category")
            .agg( 
                F.avg("ShockIndex").alias("Avg_ShockIndex"),
                F.avg("ModifiedShockIndex").alias("Avg_ModifiedShockIndex"),
                (F.avg("AgeShockIndex")/100).alias("Avg_AgeShockIndex_Norm"),
                F.avg("DiastolicShockIndex").alias("Avg_DiastolicShockIndex"),
                F.avg("PulsePressureIndex").alias("Avg_PulsePressureIndex"),
            ).toPandas().to_dict(orient="records"))
        
    
    return {
        "data": _cache['avg_index'] 
    }

@router_clinic_query.get("/clinic/top_cardiac_stress")
def get_top_cardiac_stress():
    return top_cardiac_stress()

def top_cardiac_stress():   
    if not 'top_cardiac_stress' in _cache:
        _cache['top_cardiac_stress'] = (
            derived_index
            .withColumn(
                "rank", 
                F.row_number()
                .over(
                    Window
                    .partitionBy("Gender")
                    .orderBy(
                        F.col("Rate_Pressure_Product").desc()
                    ) 
                )
            )
            .filter(F.col("rank") <= 5)
            .toPandas().to_dict(orient="records")
        )
    
    return {
        "data": _cache['top_cardiac_stress']
    }

@router_clinic_query.get("/clinic/obesity_mismatch")
def get_obesity_mismatch():
    return obesity_mismatch()

def obesity_mismatch(): 
    
    if not 'obesity_mismatch_query' in _cache:
        
        _cache['obesity_mismatch_query'] = (
            derived_index
            .filter(
                (F.col("Derived_BMI") > 35) & 
                (F.col("ModifiedShockIndex") < 0.7)
            )
            .select("Patient ID", "Derived_BMI", "ModifiedShockIndex", "Heart Rate", "Risk Category")
            .limit(5)
            .toPandas().to_dict(orient="records")
        )
    return {
        "data": _cache['obesity_mismatch_query']
    }

@router_clinic_query.get("/clinic/occult_shock")
def get_occult_shock():
    return occult_shock()

def occult_shock():
    if not 'occult_shock' in _cache:
        _cache['occult_shock'] = (
            derived_index
            .filter(
                (F.col("Age") < 40) & (F.col("ShockIndex") >= 0.9)
            )
            .select("Patient ID", "Age", "ShockIndex", "Body Temperature", "Risk Category")
            .limit(5)
            .toPandas().to_dict(orient="records")
        )

    return {
        "data": _cache['occult_shock']
    }

@router_clinic_query.get("/clinic/k_nearest")
def get_k_nearest(fraction: float = 0.05, radius: float = 10.0):
  
    if not 'k_nearest' in _cache:
        high_risk_center = (
            ds
            .filter(F.col("Risk Category") == "High Risk")
            .agg(
                F.median("Derived_MAP").alias("map")
                , F.median("Derived_BMI").alias("bmi")
            )
            .collect()[0]
        )
    
        _cache['k_nearest'] = (
            ds
            .filter(F.col("Risk Category") == "Low Risk")
            .withColumn(
                "Dist",
                F.sqrt(F.pow(F.col("Derived_MAP") - high_risk_center['map'], 2) 
                + 
                F.pow(F.col("Derived_BMI") - high_risk_center['bmi'], 2))
            ) 
            .filter(
                (F.col("Dist") < radius) & (F.col("Dist") >= 0.01)
            )
            .orderBy("Dist")
            .select("Derived_MAP", "Derived_BMI", "Risk Category")
            .sample(withReplacement=False, fraction=fraction, seed=42)
            .toPandas().to_dict(orient="records")
            + [
                {
                    "Derived_MAP": high_risk_center['map'],
                    "Derived_BMI": high_risk_center['bmi'], 
                    "Risk Category": "High Risk"
                }
            ]
        )
    return {
        "data": _cache['k_nearest']
    }

@router_clinic_query.get("/clinic/metabolic_shockindex")
def get_metabolic_shockindex(fraction: float = 0.05):
    
    if not 'metabolic_shockindex' in _cache:
        _cache['metabolic_shockindex'] = (
            derived_index
            .select(
                "ShockIndex",
                "PulsePressureIndex",
                "Risk Category"
            )
            .sample(withReplacement=False, fraction=fraction, seed=42)
            .toPandas().to_dict(orient="records")
        )
    return {
        "data": _cache['metabolic_shockindex']
    }