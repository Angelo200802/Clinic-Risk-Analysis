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

@router_clinic_query.get("/clinic/shockindex")
def get_shock_index():
    shock_index_df = (
        ds
        .withColumn("ShockIndex", F.col("Heart Rate") / F.col("Systolic Blood Pressure"))
        .withColumn("ModifiedShockIndex", F.col("Heart Rate") / F.col("Derived_MAP"))
        .withColumn("AgeShockIndex", F.col("ShockIndex") * F.col("Age"))
        .withColumn("DiastolicShockIndex", F.col("Heart Rate") / F.col("Diastolic Blood Pressure"))
    )
    shock_index_stats = shock_index_df.select(
        F.col("Patient ID"), 
        F.col("Heart Rate"), 
        F.col("Systolic Blood Pressure"), 
        F.col("ShockIndex"),
        F.col("ModifiedShockIndex"), 
        F.col("AgeShockIndex"), 
        F.col("DiastolicShockIndex")
    )

    return {
        "data": shock_index_stats.toPandas().head(5).to_dict(orient="records")
    }


@router_clinic_query.get("/clinic/roxindex")
def get_rox_index():
    rox_index_df = ds.withColumn("ROXIndex", F.col("Oxygen Saturation") / F.col("Respiratory Rate"))
    rox_index_stats = rox_index_df.select(
        F.col("Patient ID"),
        F.col("Oxygen Saturation"), 
        F.col("Respiratory Rate"),
        F.col("ROXIndex"),
    )

    return {
        "data": rox_index_stats.toPandas().head(5).to_dict(orient="records")    
    }


@router_clinic_query.get("/clinic/pulsepressureindex")
def get_pulse_pressure_index():
    ppi_df = ds.withColumn("PulsePressureIndex", F.col("Derived_Pulse_Pressure") / F.col("Heart Rate"))
    ppi_stats = ppi_df.select(
        F.col("Patient ID"), 
        F.col("Systolic Blood Pressure"), 
        F.col("Diastolic Blood Pressure"), 
        F.col("Derived_Pulse_Pressure"), 
        F.col("Heart Rate"), 
        F.col("PulsePressureIndex")
    )

    return {
        "data": ppi_stats.toPandas().head(5).to_dict(orient="records")    
    }

@router_clinic_query.get("/clinic/shockpatients")
def get_shock_patient(order: str = "desc"):
    shock_df = ds \
        .withColumn(
            "ShockIndex", 
            F.col("Heart Rate") / F.col("Systolic Blood Pressure")) \
        .filter(F.col("ShockIndex") > 0.85) \
        .orderBy(F.col("ShockIndex").asc() if order == "asc" else F.col("ShockIndex").desc() ) \
        .select("Patient ID","Heart Rate", "Systolic Blood Pressure", "Prediction", "ShockIndex")
    pd_shock = shock_df.toPandas()  
    logging.info(f"Number of patients with Shock Index > 0.9: {len(pd_shock)}") 
    return {
        "data": pd_shock.head(5).to_dict(orient="records"),
        "count": len(pd_shock),   
    }

@router_clinic_query.get("/clinic/hemodynamicrisk")
def get_hemodynamic_risk():
    hemodynamic_risk = ds.withColumn("Shock_Index", F.col("Heart Rate") / F.col("Systolic Blood Pressure")) \
    .filter((F.col("Derived_MAP") < 70) & (F.col("Shock_Index") > 0.9)) \
    .select("Patient ID", "Derived_MAP", "Shock_Index", "Risk Category")

    return {
        "data": hemodynamic_risk.toPandas().head(5).to_dict(orient="records")
    }

@router_clinic_query.get("/clinic/ratepressure_product")
def get_rate_pressure_product():
    rate_pressure_product = ds.withColumn("Rate_Pressure_Product", F.col("Heart Rate") * F.col("Systolic Blood Pressure"))
    return {
        "data": rate_pressure_product.select("Patient ID", "Heart Rate", "Systolic Blood Pressure", "Rate_Pressure_Product").toPandas().head(5).to_dict(orient="records")
    }

@router_clinic_query.get("/clinic/metabolic_effs")
def get_metabolic_effects():
    metabolic_stress = ds.withColumn("Cardiac_Effort", F.col("Heart Rate") * F.col("Derived_BMI")) \
    .groupBy("Risk Category") \
    .agg(F.avg("Cardiac_Effort").alias("Sforzo_Metabolico_Medio"))

    return {
        "data": metabolic_stress.toPandas().to_dict(orient="records")
    }