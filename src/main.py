from fastapi import FastAPI
from query.clinic_query import router
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.include_router(router)

@app.get("/")
def hello_world():
    return {"message": "Hello World from Vital Signs Analysis Application!"}


if __name__ == "__main__":
    pass
    