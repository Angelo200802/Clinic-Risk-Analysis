from pyspark.sql.types import StructType, StructField,IntegerType, DoubleType, StringType
from pyspark.sql.functions import col
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
    StructField("Patient ID", StringType(), True),
    StructField("Heart Rate", StringType(), True),
    StructField("Respiratory Rate", StringType(), True),
    StructField("Timestamp", StringType(), True),
    StructField("Body Temperature", StringType(), True),
    StructField("Oxygen Saturation", StringType(), True),
    StructField("Systolic Blood Pressure", StringType(), True),
    StructField("Diastolic Blood Pressure", StringType(), True),
    StructField("Age", StringType(), True),
    StructField("Gender", StringType(), True),
    StructField("Weight (kg)", StringType(), True),
    StructField("Height (m)", StringType(), True),
    StructField("Derived_HRV", StringType(), True),
    StructField("Derived_Pulse_Pressure", StringType(), True),
    StructField("Derived_BMI", StringType(), True),
    StructField("Derived_MAP", StringType(), True),
    StructField("Risk Category", StringType(), True)
])

def batch_job(df_batch, batch_id):
    df_batch.cache() 
    
    current_count = df_batch.count()
    
    if current_count > 0:
        logging.info(f"--- START BATCH {batch_id} (Records: {current_count}) ---")
        df_batch.show(truncate=False)
        try:
            df_cleaned = df_batch.select([
                col(c).cast("int") if "ID" in c or "Age" in c or "Rate" in c else 
                col(c).cast("double") if "Body" in c or "Oxygen" in c or "Pressure" in c or "Weight" in c or "Height" in c or "Derived" in c else 
                col(c).cast("timestamp") if "Timestamp" in c else
                col(c) for c in df_batch.columns
            ])
            
            prediction = ensemble.classify(df_cleaned).collect()
            logging.info(f"Batch {batch_id} completato. Predizioni: {len(prediction)}")
            #send results via websocket
            logging.info(f"--- END BATCH {batch_id} ---")
        except Exception as e:
            logging.error(f"Errore nel processamento del batch: {e}")
    
    df_batch.unpersist()

def start_streaming():
    df_stream = (
        get_session().readStream 
            .format("redis") 
            .option("redis.host", os.getenv("REDIS_HOST","redis"))  
            .option("redis.port", os.getenv("REDIS_PORT", "6379"))    
            .option("stream.keys", "vital_signs") 
            .option("stream.read.batch.size", "50") 
            .option("stream.group.name", f"spark-group-{int(time.time())}")
            .schema(schema)
            .load()
        )

    if os.path.exists("/tmp/spark_checkpoint"):
        import shutil
        shutil.rmtree("/tmp/spark_checkpoint")

    _streaming_query = df_stream.writeStream \
        .foreachBatch(batch_job) \
        .option("checkpointLocation", "/tmp/spark_checkpoint") \
        .start()
    
    return _streaming_query

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

