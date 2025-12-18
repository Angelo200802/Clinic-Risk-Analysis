from pyspark.sql import SparkSession
from dotenv import load_dotenv
import os

def main():
    # Initialize Spark session
    load_dotenv()
    ds_path = os.getenv("DATASET_PATH")
    print(f"Loading dataset from: {ds_path}")
    spark = (
        SparkSession.builder
        .appName("VitalSignsProject")
        .master("local[*]")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )

    df = spark.read.csv(ds_path, header=True, inferSchema=True)

    print("Schema of the dataset:")
    print("First 5 rows of the dataset:")
    df.show(5)

if __name__ == "__main__":
    main()