from fastapi import FastAPI
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
def hello_world():
    return {"message": "Hello World from Vital Signs Analysis Application!"}


if __name__ == "__main__":
    logging.info("Starting Vital Signs Analysis Application")
    