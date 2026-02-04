from pyspark.sql import SparkSession
from model.ensemble import Ensemble 
from dotenv import load_dotenv
import logging, os

load_dotenv()
_spark = None
_dataset = None

def batch_job(df_batch, batch_id):
    if df_batch.count() > 0:
        logging.info(f"Processing batch id: {batch_id}")
        df_batch.show(truncate=False)
def start_streaming():
    df_stream = (
        get_session().readStream 
            .format("redis") 
            .option("redis.host", os.getenv("REDIS_HOST","redis"))  
            .option("redis.port", os.getenv("REDIS_PORT", "6379"))    
            .option("stream.keys", "vital_signs") 
            .option("stream.read.batch.size", "50") 
            .load()
        )

    _streaming_query = df_stream.writeStream \
        .foreachBatch(batch_job) \
        .option("checkpointLocation", "/tmp/spark_checkpoint") \
        .start()

def get_session() -> SparkSession:
    logging.info("Initializing Spark session")
    redis_package = "com.redislabs:spark-redis_2.12:2.4.2"
    global _spark
    if _spark:
        return _spark
    _spark = (
        SparkSession.builder
        .appName("VitalSignsProject")
        .master("local[*]")
        #.config("spark.jars.packages", redis_package) # Scarica il connettore automaticamente
        .config("spark.driver.memory", "4g")          
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "2")  
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