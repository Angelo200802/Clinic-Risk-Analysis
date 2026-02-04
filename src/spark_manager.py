from pyspark.sql import SparkSession
from model.ensemble import Ensemble 
from dotenv import load_dotenv
import logging, os

load_dotenv()
_spark = None
_dataset = None


def get_session() -> SparkSession:
    logging.info("Initializing Spark session")
    global _spark
    if _spark:
        return _spark
    _spark = (
        SparkSession.builder
        .appName("VitalSignsProject")
        .master("local[*]")
        #.config("spark.jars.packages", redis_package) # Scarica il connettore automaticamente
        .config("spark.driver.memory", "4g")   
        .config("spark.ui.enabled", "false")
        .config("spark.port.maxRetries", "100")       
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "2")  
        .getOrCreate()
    )

    return _spark

def load_dataset(ds_path: str):
    if _spark is None:
        get_session()
    model = Ensemble()
    global _dataset
    if not _dataset:
        logging.info(f"Loading dataset from: {ds_path}")
        _dataset = model.classify(_spark.read.csv(ds_path, header=True, inferSchema=True))
    
    return _dataset