from pyspark.sql.types import StructType, StructField,IntegerType, TimestampType, DoubleType, StringType
from pyspark.sql.functions import col, window, avg, max, min
from redis import Redis
from fastapi import APIRouter
from spark_manager import load_dataset, get_session
import os, logging, time
from model.ensemble import Ensemble
from pydantic import BaseModel, Field
from dotenv import load_dotenv  

load_dotenv()
logging.basicConfig(level=logging.INFO)

redis_db = Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT")), db=0)
ensemble = Ensemble()
router_streaming = APIRouter()

class VitalSigns(BaseModel):
    patient_id : int = Field(alias="Patient ID")
    heart_rate : int = Field(alias="Heart Rate")
    respiratory_rate : int = Field(alias="Respiratory Rate")
    timestamp : str = Field(alias="Timestamp")
    body_temperature : float = Field(alias="Body Temperature")
    oxygen_saturation : float = Field(alias="Oxygen Saturation")
    systolic_blood_pressure : float = Field(alias="Systolic Blood Pressure")
    diastolic_blood_pressure : float = Field(alias="Diastolic Blood Pressure")
    age : int = Field(alias="Age")
    gender : str = Field(alias="Gender")
    weight_kg : float = Field(alias="Weight (kg)")
    height_m : float = Field(alias="Height (m)")
    derived_hrv : float = Field(alias="Derived_HRV")
    derived_pulse_pressure : float = Field(alias="Derived_Pulse_Pressure")
    derived_bmi : float = Field(alias="Derived_BMI")
    derived_map : float = Field(alias="Derived_MAP")
    risk_category : str = Field(default= None, alias="Risk Category")
    
df = load_dataset(os.getenv("DATASET_PATH"))

schema = StructType([
    StructField("Patient ID", IntegerType(), True),
    StructField("Heart Rate", IntegerType(), True),
    StructField("Respiratory Rate", IntegerType(), True),
    StructField("Timestamp", StringType(), True),
    StructField("Body Temperature", DoubleType(), True),
    StructField("Oxygen Saturation", DoubleType(), True),
    StructField("Systolic Blood Pressure", DoubleType(), True),
    StructField("Diastolic Blood Pressure", DoubleType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("Weight (kg)", DoubleType(), True),
    StructField("Height (m)", DoubleType(), True),
    StructField("Derived_HRV", DoubleType(), True),
    StructField("Derived_Pulse_Pressure", DoubleType(), True),
    StructField("Derived_BMI", DoubleType(), True),
    StructField("Derived_MAP", DoubleType(), True),
    StructField("Risk Category", StringType(), True)
])

def batch_job(df_batch, batch_id):
    df_batch.cache() 
    
    current_count = df_batch.count()
    
    if current_count > 0:
        logging.info(f"--- START BATCH {batch_id} (Records: {current_count}) ---")
        try:
            prediction = ensemble.classify(df_batch).collect()
            logging.info(f"Batch {batch_id} completato. Predizioni: {len(prediction)}")
            #send results via websocket
            logging.info(f"--- END BATCH {batch_id} ---")
        except Exception as e:
            logging.error(f"Errore nel processamento del batch: {e}")
    
    df_batch.unpersist()

def batch_job_stats(df_stats, batch_id):
    logging.info(f"--- START STATS WINDOW {batch_id} ---")   
    df_stats.show()
    count = df_stats.count()
    if count > 0:
        logging.info(f"Finestra stats batch {batch_id} con {count} record.")
        ds_stats = df_stats.collect()  
        for row in ds_stats:
            logging.info(f"Finestra: {row}")

def start_streaming():
    df_stream = (
        get_session().readStream 
            .format("redis") 
            .option("redis.host", os.getenv("REDIS_HOST","redis"))  
            .option("redis.port", os.getenv("REDIS_PORT", "6379"))    
            .option("stream.keys", "vital_signs") 
            .option("stream.read.batch.size", "50") 
            .option("stream.group.name", f"spark-classification")
            .schema(schema)
            .load()
        )
    df_stats_raw = (get_session().readStream 
        .format("redis") 
        .option("stream.keys", "vital_signs") 
        .option("stream.group.name", "spark-statistics")     # Gruppo B
        .schema(schema)
        .load())
    
    df_stats_raw = df_stats_raw.withColumn("Timestamp", col("Timestamp").cast("timestamp"))

    classification_query = (df_stream.writeStream 
        .foreachBatch(batch_job) 
        .option("checkpointLocation", "/tmp/spark_checkpoint") 
        .start())


    df_windowed = (df_stats_raw
        .withWatermark("Timestamp", "1 minute")
        .groupBy(
            window(col("Timestamp"), "1 minute", "30 seconds"),
            col("Patient ID")
        )
        .agg(
            avg("Heart Rate").alias("avg_heart_rate"),
            min("Heart Rate").alias("min_heart_rate"),
            max("Heart Rate").alias("max_heart_rate"),
            avg("Respiratory Rate").alias("avg_respiratory_rate"),
        )
    )

    query_stats = (df_windowed.writeStream
        .foreachBatch(batch_job_stats)
        .outputMode("update")
        .option("truncate", "false")
        .start()
    )

    return [ classification_query,query_stats]

@router_streaming.get("/getseed")
def get_seed():
    seed = df.rdd.takeSample(False, 1)[0].asDict()
    ret = {}
    for key in seed:
        if key not in ['Risk Category','pred_logistic_regression','pred_mlp', 'pred_naive_bayes', 'weighted_score', 'Prediction'] : 
            ret[key] = seed[key]
    return ret

@router_streaming.post("/newraw")
def new_raw(raw: VitalSigns):
    raw_dict = raw.model_dump(by_alias=True, exclude_none=False)
    clean_data = {k: (v if v is not None else "") for k, v in raw_dict.items()}
    redis_db.xadd('vital_signs', clean_data)
    
    return {"status": "ok"}

