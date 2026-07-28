from contextlib import asynccontextmanager
from fastapi import FastAPI
from pyspark.ml import PipelineModel
from query.clinic_query import (
    router_clinic_query,
    derived_indices,
    top_cardiac_stress,
    obesity_mismatch,
    occult_shock
)
from query.model_evaluation import (
    router_model_ev, 
    evaluation_by_category,
    metrics,
    metrics_shock_risk,
    confusion_matrix,
    ensemble_consensus
)
from query.stats import (
    router_stats,
    correlation_matrix,
    risk_composition,
    get_demographic_stress_map,
    gender_risk,
    bmi_risk,
    age_risk
)
from streaming import router_streaming, start_streaming
import logging, dotenv, asyncio
import bus
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up the Vital Signs Analysis Application...")
    bus.main_loop = asyncio.get_event_loop()
    spark_query = start_streaming()
    logging.info("Precomputing query")
    evaluation_by_category()
    metrics()
    metrics_shock_risk()
    confusion_matrix()
    ensemble_consensus() 
    correlation_matrix()
    risk_composition()
    get_demographic_stress_map()
    gender_risk()
    bmi_risk()
    age_risk()
    derived_indices(),
    top_cardiac_stress(),
    obesity_mismatch(),
    occult_shock()
    logging.info("Application is ready to serve requests.")
    yield
    logging.info("Shutting down the Vital Signs Analysis Application...")
    for q in spark_query: q.stop()

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(lifespan=lifespan)
app.include_router(router_stats)
app.include_router(router_clinic_query)
app.include_router(router_model_ev)
app.include_router(router_streaming)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def hello_world():
    return {"message": "Hello World from Vital Signs Analysis Application!"}

    