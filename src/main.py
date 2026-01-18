from fastapi import FastAPI
from query.clinic_query import router_clinic_query
from query.model_evaluation import router_model_ev
from query.stats import router_stats
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.include_router(router_stats)
app.include_router(router_clinic_query)
app.include_router(router_model_ev)

@app.get("/")
def hello_world():
    return {"message": "Hello World from Vital Signs Analysis Application!"}


if __name__ == "__main__":
    logging.log(logging.INFO, "Starting Vital Signs Analysis Application...")
    