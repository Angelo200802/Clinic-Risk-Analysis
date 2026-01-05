from src.main import app
from spark_manager import get_session, load_dataset

spark = get_session()

@app.get("/stats/{signs}")
def get_stats(signs:str):
    return {}