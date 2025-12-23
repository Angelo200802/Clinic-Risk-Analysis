from pyspark.sql import SparkSession    
from pyspark.ml.classification import LogisticRegression
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import logging, os

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

lr = LogisticRegression(maxIter=10,regParam=0.3,elasticNetParam=0.8)

lr.fit(df)