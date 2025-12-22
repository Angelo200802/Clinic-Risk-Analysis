from pyspark.sql import SparkSession
from dotenv import load_dotenv
import os, logging

logging.basicConfig(level=logging.INFO)

def main():
    # Initialize Spark session
    load_dotenv()
    ds_path = os.getenv("DATASET_PATH")
    spark = (
        SparkSession.builder
        .appName("VitalSignsProject")
        .master("local[*]")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )
    logging.info(f"Loading dataset from: {ds_path}")
    df = spark.read.csv(ds_path, header=True, inferSchema=True)
    
    logging.info("First 5 rows of the dataset:")
    df.show(5)

if __name__ == "__main__":
    logging.info("Starting Vital Signs Analysis Application")
    main()