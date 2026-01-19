import os
from fastapi import FastAPI
from spark_manager import load_dataset
from query.clinic_query import router_clinic_query
from query.model_evaluation import router_model_ev
from query.stats import router_stats
import logging, dotenv,os

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

df = load_dataset(os.getenv("DATASET_PATH"))
app = FastAPI()
app.include_router(router_stats)
app.include_router(router_clinic_query)
app.include_router(router_model_ev)

@app.get("/")
def hello_world():
    return {"message": "Hello World from Vital Signs Analysis Application!"}

@app.get("/getseed")
def get_seed():
    return df.rdd.takeSample(False, 1)[0].asDict()

@app.post("/newraw")
def new_raw(raw: dict):
    logging.log(logging.INFO,f"Received new raw data: {raw}")

if __name__ == "__main__":
    logging.log(logging.INFO, "Starting Vital Signs Analysis Application...")
    