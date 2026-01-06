from pyspark.sql import SparkSession
import logging

_spark = None

def get_session() -> SparkSession:

    logging.info("Initializing Spark session")
    global _spark
    _spark = (
        SparkSession.builder
        .appName("VitalSignsProject")
        .master("local[*]")
        .config("spark.driver.memory", "4g")    # Assegna 4GB al driver
        .config("spark.executor.memory", "4g")  # Assegna 4GB agli esecutori
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )

    return _spark

def load_dataset(ds_path: str):
    logging.info(f"Loading dataset from: {ds_path}")
    df = _spark.read.csv(ds_path, header=True, inferSchema=True)
    
    return df