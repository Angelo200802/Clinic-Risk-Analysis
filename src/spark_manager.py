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
    global _dataset
    if _spark is None:
        get_session()

    parquet_path = ds_path.replace(".csv", "_classified.parquet")
    
    if _dataset:
        return _dataset

    if os.path.exists(parquet_path):
        logging.info(f"File Parquet trovato. Caricamento in corso: {parquet_path}")
        _dataset = _spark.read.parquet(parquet_path).cache()
    else:
        logging.info(f"Parquet non trovato. Caricamento e classificazione CSV: {ds_path}")
        model = Ensemble()
        
        raw_df = _spark.read.csv(ds_path, header=True, inferSchema=True)
        _dataset = model.classify(raw_df).cache()
        
        try:
            logging.info(f"Salvataggio dataset classificato in: {parquet_path}")
            _dataset.write.mode("overwrite").parquet(parquet_path)
            logging.info("Salvataggio completato con successo.")
        except Exception as e:
            logging.error(f"Errore durante il salvataggio del Parquet: {e}")

    logging.info("Dataset Columns: " + ", ".join(_dataset.columns))
    _dataset.show(5)
    return _dataset