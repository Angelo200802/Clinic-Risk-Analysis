from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from redis import Redis
from fastapi import APIRouter
from spark_manager import load_dataset
import os, logging
from pydantic import BaseModel, Field
from dotenv import load_dotenv  

load_dotenv()
logging.basicConfig(level=logging.INFO)

redis_db = Redis(host='redis', port=6379, db=0)

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
    derived_hrv : float = Field(alias="Derived HRV")
    derived_pulse_pressure : float = Field(alias="Derived Pulse Pressure")
    derived_bmi : float = Field(alias="Derived BMI")
    derived_map : float = Field(alias="Derived MAP")
    risk_category : str = Field(default= None, alias="Risk Category")

df = load_dataset(os.getenv("DATASET_PATH"))
@router_streaming.get("/getseed")
def get_seed():
    return df.rdd.takeSample(False, 1)[0].asDict()

@router_streaming.post("/newraw")
def new_raw(raw: VitalSigns):
    logging.log(logging.INFO,f"Received new raw data: {raw}")
    redis_db.xadd('vital_signs', raw.model_json_schema())

schema = StructType([
    StructField("Heart_Rate", DoubleType(), True),
    StructField("Systolic_BP", DoubleType(), True),
    StructField("Diastolic_BP", DoubleType(), True),
    StructField("SpO2", DoubleType(), True),
    StructField("Body_Temperature", DoubleType(), True),
    StructField("Respiratory_Rate", DoubleType(), True)
])