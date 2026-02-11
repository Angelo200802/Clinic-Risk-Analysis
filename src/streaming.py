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
            .withColumn("prev_avg_hr_lag", F.lag("avg_hr", 2).over(patient_window))
            .withColumn("prev_avg_rr", F.lag("avg_rr").over(patient_window))
            .withColumn("prev_avg_rr_lag", F.lag("avg_rr", 2).over(patient_window))
            .withColumn("prev_avg_map", F.lag("avg_map").over(patient_window))
            .withColumn("prev_avg_map_lag", F.lag("avg_map", 2).over(patient_window))
            .withColumn("prev_avg_spo2", F.lag("avg_spo2").over(patient_window))
            .withColumn("prev_avg_spo2_lag", F.lag("avg_spo2", 2).over(patient_window))
            .withColumn("prev_avg_hrv", F.lag("avg_hrv").over(patient_window))
            .withColumn("prev_avg_hrv_lag", F.lag("avg_hrv", 2).over(patient_window))
            .withColumn("prev_avg_pp", F.lag("avg_pp").over(patient_window))
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
                "hr_pct",
                F.when(
                    (F.col("prev_avg_hr").isNull()) | (F.col("prev_avg_hr") == 0),
                    None
                ).otherwise(
                    (F.col("avg_hr") - F.col("prev_avg_hr")) / F.col("prev_avg_hr") * 100
                )
            )
            .withColumn(
                "rr_pct",
                F.when(
                    (F.col("prev_avg_rr").isNull()) | (F.col("prev_avg_rr") == 0),
                    None
                ).otherwise(
                    (F.col("avg_rr") - F.col("prev_avg_rr")) / F.col("prev_avg_rr") * 100
                )
            )   
            .withColumn(
                "spo2_pct",
                F.when(
                    (F.col("prev_avg_spo2").isNull()) | (F.col("prev_avg_spo2") == 0),
                    None
                ).otherwise(
                    (F.col("avg_spo2") - F.col("prev_avg_spo2")) / F.col("prev_avg_spo2") * 100
                )
            )
            .withColumn(
                "pp_pct",
                F.when(
                    (F.col("prev_avg_pp").isNull()) | (F.col("prev_avg_pp") == 0),
                    None
                ).otherwise(
                    (F.col("avg_pp") - F.col("prev_avg_pp")) / F.col("prev_avg_pp") * 100
                )
            )
            .withColumn(
                "map_pct",
                F.when(
                    (F.col("prev_avg_map").isNull()) | (F.col("prev_avg_map") == 0),
                    None
                ).otherwise(
                    (F.col("avg_map") - F.col("prev_avg_map")) / F.col("prev_avg_map") * 100
                )
            )
            .withColumn(
                "progressive_hemo_deterioration",
                F.when(
                    (F.col("avg_hr") > F.col("prev_avg_hr")) &
                    (F.col("prev_avg_hr") > F.col("prev_avg_hr_lag")) &
                    (F.col("avg_map") < F.col("prev_avg_map")) &
                    (F.col("prev_avg_map") < F.col("prev_avg_map_lag")),
                    1
                ).otherwise(0)
            )
            .withColumn(
                "progressive_resp_failure_pattern",
                F.when(
                    (F.col("avg_rr") > F.col("prev_avg_rr")) &
                    (F.col("prev_avg_rr") > F.col("prev_avg_rr_lag")) &
                    (F.col("avg_spo2") < F.col("prev_avg_spo2")) &
                    (F.col("prev_avg_spo2") < F.col("prev_avg_spo2_lag")),
                    1
                ).otherwise(0)
            )
            .withColumn(
                "dynamic_sepsis_pattern",
                F.when(
                    (F.col("avg_temp") > 38) &
                    (F.col("avg_hr") > F.col("prev_avg_hr")) &
                    (F.col("prev_avg_hr") > F.col("prev_avg_hr_lag")) &
                    (F.col("avg_hrv") < F.col("prev_avg_hrv")) &
                    (F.col("prev_avg_hrv") < F.col("prev_avg_hrv_lag")),
                    1
                ).otherwise(0)
            )
            .withColumn(
                "shock_index",
                F.col("avg_hr") / F.col("avg_sbp")
            )
            .withColumn(
                "modified_shock_index",
                F.col("avg_hr") / F.col("avg_map")
            )
            .withColumn(
                "age_index",
                F.col("Age") * F.col("shock_index")
            )
            .withColumn(
                "diastolic_shock_index",
                F.col("avg_hr") / F.col("avg_dbp")
            )
            .withColumn(
                "rate_pp",
                F.col("avg_sbp") * F.col("avg_hr")
            )
            .withColumn(
                "pp_index",
                F.col("avg_pp") / F.col("avg_hr")
            )
            .withColumn(
                "rox_index",
                F.col("avg_spo2") / F.col("avg_rr")
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
                        k : v for k, v in dict_row.items() 
                        if k not in columns 
                        and 'prev' not in k 
                        and 'index' not in k
                        and 'pattern' not in k
                        and 'rate_pp' not in k
                        and 'delta' not in k
                    } ,
                    "index" : {
                        k : v for k, v in dict_row.items() 
                        if 'index' in k 
                        or 'rate_pp' in k
                    } ,
                    "pattern" :{
                        k : v for k, v in dict_row.items() 
                        if 'pattern' in k or 'deterioration' in k
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
            F.avg("Systolic Blood Pressure").alias("avg_sbp"),
            F.avg("Diastolic Blood Pressure").alias("avg_dbp"),
            F.avg("Respiratory Rate").alias("avg_rr"),
            F.avg("Oxygen Saturation").alias("avg_spo2"),
            F.avg("Body Temperature").alias("avg_temp"),
            F.avg("Derived_MAP").alias("avg_map"),
            F.avg("Derived_Pulse_Pressure").alias("avg_pp"),
            F.avg("Derived_HRV").alias("avg_hrv"),
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

    return [query_stats]

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