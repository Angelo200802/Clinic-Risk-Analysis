import os
from fastapi import FastAPI
from spark_manager import load_dataset
from query.clinic_query import router_clinic_query
from query.model_evaluation import router_model_ev
from query.stats import router_stats
from streaming import router_streaming
import logging, dotenv,os

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

df = load_dataset(os.getenv("DATASET_PATH"))
app = FastAPI()
app.include_router(router_stats)
app.include_router(router_clinic_query)
app.include_router(router_model_ev)
app.include_router(router_streaming)

@app.get("/")
def hello_world():
    return {"message": "Hello World from Vital Signs Analysis Application!"}

if __name__ == "__main__":
    logging.log(logging.INFO, "Starting Vital Signs Analysis Application...")
    