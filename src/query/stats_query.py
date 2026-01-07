from src.main import app
from src.spark_manager import get_session, load_dataset
from src.model.ensemble import Ensemble
import os

SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

spark = get_session()
ds = load_dataset(os.getenv("DATASET_PATH"))
ensemble_model = Ensemble(models={
    "logistic_regression": SAVE_MODEL_PATH+"/log_reg_pipeline",
    "naive_bayes": SAVE_MODEL_PATH+"/nb_pipeline",
    "mlp" : SAVE_MODEL_PATH+"/ann_model_pipeline"
})


@app.get("/stats/{signs}")
def get_stats(signs:str):
    return {}
