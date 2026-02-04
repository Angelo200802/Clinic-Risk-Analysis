from pyspark.sql import SparkSession
from model.ensemble import Ensemble 
import logging

_spark = None
_dataset = None

def batch_job(df_batch, batch_id):
    if df_batch.count() > 0:
        logging.info(f"Processing batch id: {batch_id}")
        df_batch.show(truncate=False)
def start_streaming():
    df_stream = _spark.readStream \
                .format("redis") \
                .option("stream.keys", "vital_signs") \
                .option("stream.read.batch.size", "50") \
                .load()

    _streaming_query = df_stream.writeStream \
        .foreachBatch(batch_job) \
        .option("checkpointLocation", "/tmp/spark_checkpoint") \
        .start()

def get_session() -> SparkSession:
    jar_path = "/app/jars/spark-redis-3.1.0-with-dependencies.jar"
    logging.info("Initializing Spark session")
    global _spark
    if _spark:
        return _spark
    _spark = (
        SparkSession.builder
        .appName("VitalSignsProject")
        .master("local[*]")
        .config("spark.driver.memory", "4g")    # Assegna 4GB al driver
        .config("spark.executor.memory", "4g")  # Assegna 4GB agli esecutori
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )

    start_streaming()

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