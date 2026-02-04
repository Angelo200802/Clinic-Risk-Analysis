from pyspark.sql.types import StructType, StructField,IntegerType, DoubleType, StringType
from redis import Redis
from fastapi import APIRouter
from spark_manager import load_dataset
import os, logging
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
df = load_dataset(os.getenv("DATASET_PATH"))

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
    logging.log(logging.INFO,f"Received new raw data: {raw}")
    redis_db.xadd('vital_signs', raw.model_dump(by_alias=True,exclude_none=True))

