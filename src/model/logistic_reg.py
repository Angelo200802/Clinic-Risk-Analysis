from src.spark_manager import get_session, load_dataset
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from dotenv import load_dotenv
import os, logging

load_dotenv()
ds_path = os.getenv("DATASET_PATH")

spark = get_session()   
ds = load_dataset(spark, ds_path)

indexer_gender = StringIndexer(
    inputCol = "Gender",
    outputCol = "Gender_b",
    stringOrderType = "alphabetAsc"
)

indexer_risk = StringIndexer(
    inputCol = ds.columns[-1],
    outputCol = "RiskCategory_b",
    stringOrderType = "alphabetAsc"
)

assembler = VectorAssembler(
    inputCols=[col for col in ds.columns if col != ds.columns[-1] and col != "Gender" and col != "Timestamp" ],
    outputCol="features"
)

lr = LogisticRegression(labelCol="RiskCategory_b", featuresCol="features")
train, test = ds.randomSplit([0.7, 0.3], seed=42)

pipe = Pipeline(stages=[indexer_gender, indexer_risk, assembler, lr])
model = pipe.fit(train)
predictions = model.transform(test) 

print("Logistic Regression Model Summary:")
lr_model = model.stages[-1]
training_summary = lr_model.summary
print(f"Accuracy: {training_summary.accuracy}")
print(f"Precision: {training_summary.weightedPrecision}")
print(f"Recall: {training_summary.weightedRecall}")
print(f"F1 Score: {training_summary.weightedFMeasure()}")   
predictions.select("features", "RiskCategory_b", "prediction").show(10)