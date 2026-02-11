from pyspark.sql.types import StructType, StructField,IntegerType, TimestampType, DoubleType, StringType
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, window
from redis import Redis
from fastapi import APIRouter
from spark_manager import load_dataset, get_session
import os, logging
from pyspark.sql import DataFrame
from model.ensemble import Ensemble
from pydantic import BaseModel, Field
from dotenv import load_dotenv 
from fastapi import WebSocket
import bus

load_dotenv()
logging.basicConfig(level=logging.INFO)

redis_db = Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT")), db=0)
ensemble = Ensemble()
router_streaming = APIRouter()


columns = [
    "Heart Rate",
    "Respiratory Rate",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Age",
    "Gender",
    "Weight (kg)",
    "Height (m)",
    "Derived_HRV",
    "Derived_Pulse_Pressure",
    "Derived_BMI",
    "Derived_MAP",
    "Prediction"
]

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

def batch_job(df_batch : DataFrame, batch_id):
    
    df_batch.cache() 
    current_count = df_batch.count()
    
    if current_count > 0:
        logging.info(f"--- START BATCH {batch_id} (Records: {current_count}) ---")
        try:
            prediction = ensemble.classify(df_batch).collect() 
            logging.info(f"Batch {batch_id} completato. Predizioni: {len(prediction)}")
            
            for row in prediction:
                data_dict = row.asDict()
                bus.main_loop.call_soon_threadsafe(
                    bus.data_queue.put_nowait, 
                    {
                        "type" : "prediction",
                        "data" : { k : data_dict[k] for k in data_dict if k not in ['Risk Category','pred_logistic_regression','pred_mlp', 'pred_naive_bayes', 'weighted_score'] }
                    }
                )
            logging.info(f"--- END BATCH {batch_id} ---")
        except Exception as e:
            logging.error(f"Errore nel processamento del batch: {e}")
    
    df_batch.unpersist()

def batch_job_stats(df_stats : DataFrame, batch_id): 
    df_stats.show()
    count = df_stats.count()
    if count > 0:
        patient_window = Window.partitionBy("Patient ID").orderBy("window.start")

        df_with_trend = (
            df_stats
            .withColumn("prev_avg_hr", F.lag("avg_hr").over(patient_window))
            .withColumn("prev_avg_map", F.lag("avg_map").over(patient_window))
            .withColumn("prev_avg_spo2", F.lag("avg_spo2").over(patient_window))
            .withColumn("prev_avg_hrv", F.lag("avg_hrv").over(patient_window))
            .withColumn("hr_delta", F.col("avg_hr") - F.col("prev_avg_hr"))
            .withColumn("map_delta", F.col("avg_map") - F.col("prev_avg_map"))
            .withColumn("spo2_delta", F.col("avg_spo2") - F.col("prev_avg_spo2"))
            .withColumn("hrv_delta", F.col("avg_hrv") - F.col("prev_avg_hrv"))
            .withColumn(
                "bmi_class",
                F.when(F.col("Derived_BMI") < 18.5, "UNDERWEIGHT")
                .when(F.col("Derived_BMI") < 25, "NORMAL")
                .when(F.col("Derived_BMI") < 30, "OVERWEIGHT")
                .otherwise("OBESE")
            )
            .withColumn(
                "hr_pct_delta",
                F.when(
                    (F.col("prev_avg_hr").isNull()) | (F.col("prev_avg_hr") == 0),
                    None
                ).otherwise(
                    (F.col("avg_hr") - F.col("prev_avg_hr")) / F.col("prev_avg_hr") * 100
                )
            )
            .withColumn(
                "shock_risk",
                F.when(
                    (F.col("hr_delta") > 5) &
                    (F.col("map_delta") < -3),
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            .withColumn(
                "resp_failure_risk",
                F.when(
                    (F.col("avg_rr") > 24) &
                    (F.col("spo2_delta") < -2),
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            .withColumn(
                "sepsis_risk",
                F.when(
                    (F.col("avg_temp") > 38) &
                    (F.col("hr_delta") > 5) &
                    (F.col("hrv_delta") < -0.1),
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            .withColumn(
                "hemo_instability",
                F.when(
                    F.col("std_hr") > 10,
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            .withColumn(
                "clinical_risk_score",
                F.col("shock_risk") * 3 +
                F.col("resp_failure_risk") * 2 +
                F.col("sepsis_risk") * 3 +
                F.col("hemo_instability")
            )
        )

        for row in df_with_trend.collect():
            dict_row = row.asDict()
            dict_row['start'] = dict_row['window'].start.isoformat()
            dict_row['end'] = dict_row['window'].end.isoformat()    
            dict_row.pop('window')
            dict_row['Timestamp'] = dict_row.pop('Timestamp').isoformat() if dict_row['Timestamp'] else ""
            update = {
                    "sensor_update" : {
                        k : v for k, v in dict_row.items() if k in columns
                    },
                    "trend_update" : {
                        k : v for k, v in dict_row.items() if k not in columns and k not in ["prev_avg_hr","prev_avg_map","prev_avg_spo2","prev_avg_hrv"]
                    }   
                }
            update['sensor_update']['Patient ID'] = dict_row['Patient ID']
            update['sensor_update']['Timestamp'] = dict_row['Timestamp']
            bus.main_loop.call_soon_threadsafe(
                bus.data_queue.put_nowait, 
                update
            )
            logging.info(f"Trend calcolato per Patient ID {row['Patient ID']}\n {update}")

        logging.info(f"--- ANALISI TREND BATCH {batch_id} ---")
        df_with_trend.select(
            "Patient ID",
            "window.start",
            "n_samples",
            "avg_hr", "hr_delta",
            "avg_map", "map_delta",
            "avg_spo2", "spo2_delta",
            "avg_hrv", "hrv_delta",
            "shock_risk",
            "resp_failure_risk",
            "sepsis_risk",
            "hemo_instability",
            "clinical_risk_score"
        ).show(truncate=False)

def start_streaming():

    df_stats_raw = (get_session().readStream 
        .format("redis")
        .option("redis.host", os.getenv("REDIS_HOST","redis"))  
        .option("redis.port", os.getenv("REDIS_PORT", "6379")) 
        .option("stream.keys", "vital_signs")
        .option("stream.read.batch.size", "50") 
        .option("stream.group.name", "spark-statistics")  
        .schema(schema)
        .load())
    
    df_stats_raw = df_stats_raw.withColumn("Timestamp", col("Timestamp").cast("timestamp"))
    df_stats_raw = ensemble.classify(df_stats_raw)

    df_windowed = (
        df_stats_raw
        .withWatermark("Timestamp", "1 minute")
        .groupBy(
            window(col("Timestamp"), "1 minute", "30 seconds"),
            col("Patient ID")
        )
        .agg(
            F.last("Heart Rate").alias("Heart Rate"),
            F.last("Respiratory Rate").alias("Respiratory Rate"),
            F.last("Oxygen Saturation").alias("Oxygen Saturation"),
            F.last("Systolic Blood Pressure").alias("Systolic Blood Pressure"),
            F.last("Diastolic Blood Pressure").alias("Diastolic Blood Pressure"),
            F.last("Body Temperature").alias("Body Temperature"),
            F.last("Age").alias("Age"),
            F.last("Gender").alias("Gender"),
            F.last("Weight (kg)").alias("Weight (kg)"),
            F.last("Height (m)").alias("Height (m)"),
            F.last("Derived_MAP").alias("Derived_MAP"),
            F.last("Derived_HRV").alias("Derived_HRV"),
            F.last("Derived_BMI").alias("Derived_BMI"),
            F.last("Derived_Pulse_Pressure").alias("Derived_Pulse_Pressure"), 
            F.last("Timestamp").alias("Timestamp"),
            F.last("Prediction").alias("Prediction"),
            F.avg(
                F.when(F.lower(F.col("Prediction")) == "high risk", 1).otherwise(0)
            ).alias("risk_ratio"),
            F.avg("Heart Rate").alias("avg_hr"),
            F.max("Heart Rate").alias("max_hr"),
            F.min("Heart Rate").alias("min_hr"),
            F.avg("Respiratory Rate").alias("avg_rr"),
            F.max("Respiratory Rate").alias("max_rr"),
            F.min("Respiratory Rate").alias("min_rr"),
            F.avg("Oxygen Saturation").alias("avg_spo2"),
            F.max("Oxygen Saturation").alias("max_spo2"),
            F.min("Oxygen Saturation").alias("min_spo2"),
            F.avg("Body Temperature").alias("avg_temp"),
            F.max("Body Temperature").alias("max_temp"),
            F.min("Body Temperature").alias("min_temp"),
            F.avg("Derived_MAP").alias("avg_map"),
            F.max("Derived_MAP").alias("max_map"),
            F.min("Derived_MAP").alias("min_map"),
            F.avg("Derived_HRV").alias("avg_hrv"),
            F.max("Derived_HRV").alias("max_hrv"),
            F.min("Derived_HRV").alias("min_hrv"),
            F.stddev("Heart Rate").alias("std_hr"),
            F.count("*").alias("n_samples")
        )
        )

    query_stats = (df_windowed.writeStream
        .foreachBatch(batch_job_stats)
        .outputMode("update")
        .option("truncate", "false")
        .start()
    )

    return [query_stats]#,classification_query]

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

@router_streaming.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted.")
    try:
        while True:
            data = await bus.data_queue.get()
            await websocket.send_json(data)
    except Exception as e:
        logging.info("WebSocket connection closed.")