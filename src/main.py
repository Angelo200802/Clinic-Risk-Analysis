import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from query.clinic_query import router_clinic_query
from query.model_evaluation import router_model_ev
from query.stats import router_stats
from streaming import router_streaming, start_streaming
import logging, dotenv,os, asyncio
import bus

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up the Vital Signs Analysis Application...")
    bus.main_loop = asyncio.get_event_loop()
    spark_query = start_streaming()
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

@app.get("/")
def hello_world():
    return {"message": "Hello World from Vital Signs Analysis Application!"}

    